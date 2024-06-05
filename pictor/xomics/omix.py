# Copyright 2022 William Ro. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===-==================================================================-=======
import os
from collections import OrderedDict
from pictor.xomics.stat_analyzers import single_factor_analysis
from roma import console
from roma import io
from roma import Nomear
from typing import List

import numpy as np



class Omix(Nomear):

  def __init__(self, features, targets, feature_labels=None, sample_labels=None,
               target_labels=None, data_name='Omix'):
    """features.shape = [n_samples, n_features]"""
    self.features = features
    self.targets = np.array(targets)

    # Sanity check
    assert len(features) == len(targets), '!! features and targets must have the same length'
    assert len(features.shape) == 2, '!! features must be a 2D array'

    self._feature_labels = feature_labels
    self._target_labels = target_labels
    self.data_name = data_name

    self.set_sample_labels(sample_labels)

  # region: Properties

  @property
  def n_samples(self): return len(self.features)

  @property
  def n_features(self): return self.features.shape[1]

  @Nomear.property()
  def feature_labels(self):
    if self._feature_labels is None:
      return [f'feature-{i + 1}' for i in range(self.features.shape[1])]
    return self._feature_labels

  @Nomear.property(local=True)
  def sample_labels(self):
    return np.array([f'{i + 1}' for i in range(self.n_samples)])

  @Nomear.property()
  def target_labels(self):
    if self._target_labels is None:
      return [f'class-{i + 1}' for i in range(len(np.unique(self.targets)))]
    return self._target_labels

  @property
  def targets_are_numerical(self):
    return len(self.target_labels) == 1

  @Nomear.property(local=True)
  def target_collection(self): return OrderedDict()

  @Nomear.property()
  def groups(self):
    """Lists of indices of samples in each class"""
    return [np.where(self.targets == i)[0] for i in np.unique(self.targets)]

  @Nomear.property()
  def omix_groups(self):
    """Returns omix 01-objects for each group"""
    return [self.duplicate(features=self.features[group],
                           targets=self.targets[group],
                           sample_labels=self.sample_labels[group])
            for group in self.groups]

  @Nomear.property()
  def single_factor_analysis_reports(self):
    """reports[n] = [(i, j, p_val, method), ...], sorted by p_val,
       here n denotes the n-th feature.
    """
    reports = []
    for n in range(self.features.shape[1]):
      features = self.features[:, n]
      groups = [features[self.targets == i]
                for i, _ in enumerate(self.target_labels)]
      reports.append(single_factor_analysis(groups))
    return reports

  @Nomear.property()
  def OLS_reports(self):
    reports = []
    for n in range(self.features.shape[1]):
      features = self.features[:, n]
      reports.append(self.calc_OLS_result(features))
    return reports

  @Nomear.property()
  def feature_mean(self): return np.mean(self.features, axis=0, keepdims=True)

  @Nomear.property()
  def feature_std(self): return np.std(self.features, axis=0, keepdims=True)

  @Nomear.property()
  def corr_matrix(self): return np.corrcoef(self.features, rowvar=False)

  @property
  def sf_method(self): return self.get_from_pocket('sf_method')

  @property
  def data_frames(self):
    import pandas as pd

    dfs = []

    # Sheet 1: Features
    index = self.sample_labels
    columns = self.feature_labels
    df = pd.DataFrame(self.features, index=index, columns=columns)
    dfs.append(df)

    # Sheet 2: Default targets
    col_label = ','.join(['Target'] + self.target_labels)
    df = pd.DataFrame(self.targets, index=index, columns=[col_label])
    dfs.append(df)

    # Sheet 3- : Target collections
    for key, (targets, target_labels) in self.target_collection.items():
      col_label = ','.join([key] + target_labels)
      df = pd.DataFrame(targets, index=index, columns=[col_label])
      dfs.append(df)

    return dfs

  # endregion: Properties

  # region: Feature Selection

  def select_features(self, method: str, **kwargs):
    method = method.lower()

    omix = self.standardize() if kwargs.get('standardize', 1) else self
    save_model = kwargs.get('save_model', False)

    if method in ('pca', ):
      from sklearn.decomposition import PCA

      n_components = kwargs.get('n_components', 10)

      model = PCA(n_components=n_components)
      feature_labels = [f'PC-{i + 1}' for i in range(n_components)]
      omix_reduced = self.duplicate(features=model.fit_transform(omix.features),
                                    feature_labels=feature_labels)
    elif method in ('lasso', ):
      from pictor.xomics.ml.lasso import Lasso
      model = Lasso(kwargs.get('verbose', 0))
      omix_reduced = model.select_features(omix, **kwargs)
    elif method in ('mrmr', ):
      from mrmr import mrmr_classif
      import pandas as pd
      k = kwargs.get('k', 10)
      selected_features = mrmr_classif(
        X=pd.DataFrame(omix.features), y=pd.Series(omix.targets), K=k)
      omix_reduced = self.get_sub_space(selected_features)
      model = selected_features
    elif method in ('pval', 'sig'):
      indices = np.argsort(
        [r[0][2] for r in self.single_factor_analysis_reports])
      k = kwargs.get('k', 10)
      selected_features = indices[:k]
      omix_reduced = self.get_sub_space(selected_features)
      model = selected_features
    elif method in ('indices', ):
      indices = kwargs.get('indices', None)
      omix_reduced = self.get_sub_space(indices)
      model = indices
    else: raise KeyError(f'!! Unknown feature selecting method "{method}"')

    if save_model: omix_reduced.put_into_pocket('sf_method', model, local=True)

    return omix_reduced

  def get_sub_space(self, indices, start_from_1=True):
    """Get sub-space of features by indices.
    indices can be (1) e.g., '1,2,3', (2) e.g., '1-3', (3) e.g, [1, 2, 3]
    """
    if isinstance(indices, str):
      if '-' in indices:
        start, end = map(int, indices.split('-'))
        indices = list(range(start, end + 1))
      elif ',' in indices:
        indices = list(map(int, indices.split(',')))

    if start_from_1: indices = [i - 1 for i in indices]
    indices = np.array(indices)

    return self.duplicate(
      features=self.features[:, indices],
      feature_labels=[self.feature_labels[i] for i in indices])

  def filter_by_name(self, keywords: List[str]):
    """Filter features by keywords in feature_labels"""
    if not isinstance(keywords, list): keywords = [keywords]

    indices = []
    for i, label in enumerate(self.feature_labels):
      for key in keywords:
        if key.lower() in label.lower():
          indices.append(i)
          continue
    return self.get_sub_space(indices, start_from_1=False)

  # endregion: Feature Selection

  # region: Visualization

  def show_in_explorer(self, title='Omix', fig_size=(5, 5),
                       ignore_warnings=True):
    from pictor.xomics import FeatureExplorer
    FeatureExplorer.explore(omix=self, title=title, fig_size=fig_size,
                            ignore_warnings=ignore_warnings)

  # endregion: Visualization

  # region: IO

  @staticmethod
  def load(file_path: str, verbose=True) -> 'Omix':
    if file_path.endswith('xlsx'):
      import pandas as pd
      data_frames = pd.read_excel(file_path, sheet_name=None)
      data_name = os.path.basename(file_path).split('.')[0]

      # (1) Read features
      df = data_frames.pop('Features')
      feature_labels = df.columns.tolist()[1:]
      sample_labels = df.index.tolist()
      features = df.values[:, 1:]

      # Find legal indices from the first row
      indices, illegal_indices = [], []
      for i in range(features.shape[1]):
        try:
          np.float32(features[0, i])
          indices.append(i)
        except:
          illegal_indices.append(i)

      features = np.array(features[:, indices], dtype=np.float32)
      if verbose:
        console.show_info(f'Illegal feature labels:')
        for i in illegal_indices:
          console.supplement(f'{feature_labels[i]}', level=2)
      feature_labels = [feature_labels[i] for i in indices]

      # (2) Read targets
      df = data_frames.pop('Targets')
      targets = df.values[:, 1:].flatten()
      target_labels = df.columns.tolist()[1].split(',')[1:]

      omix = Omix(features, targets, feature_labels=feature_labels,
                  sample_labels=sample_labels, target_labels=target_labels,
                  data_name=data_name)

      # Read target collections
      for key, df in data_frames.items():
        # Consistent with the format in FeatureExplorer.save()
        if not key.startswith('Target-Collection-'): continue
        targets = df.values[:, 1:].flatten()
        mass = df.columns.tolist()[1].split(',')
        target_key = mass[0]
        target_labels = mass[1:]
        omix.add_to_target_collection(target_key, targets, target_labels)

      return omix

    return io.load_file(file_path, verbose=verbose)

  def save(self, file_path: str, verbose=True):
    if not file_path.endswith('.omix'): file_path += '.omix'
    return io.save_file(self, file_path, verbose=verbose)

  # endregion: IO

  # region: Public Methods

  def calc_OLS_result(self, x, print_summary=False):
    import statsmodels.api as sm
    x = sm.add_constant(x)
    model = sm.OLS(self.targets, x)
    result = model.fit()
    if print_summary: print(result.summary())
    return result

  def add_to_target_collection(self, key, targets, target_labels=None):
    # Sanity check
    targets = np.array(targets)
    assert len(targets) == self.n_samples, '!! targets must have the same length as samples'

    self.target_collection[key] = (targets, target_labels)

  def set_targets(self, key, return_new_omix=True):
    targets, target_labels = self.target_collection[key]

    if return_new_omix:
      return self.duplicate(targets=targets, target_labels=target_labels)

    self.targets = targets
    self._target_labels = target_labels
    self.get_from_pocket('target_labels', put_back=False)
    self.get_from_pocket('groups', put_back=False)
    self.get_from_pocket('omix_groups', put_back=False)

  def set_sample_labels(self, sample_labels):
    if sample_labels is None:
      sample_labels = np.array(
        [f'{i + 1}' for i in range(self.n_samples)])
    assert len(sample_labels) == self.n_samples
    self.put_into_pocket('sample_labels', sample_labels, local=True,
                         exclusive=False)

  def report(self, **kwargs):
    console.show_info(f'Details of `{self.data_name}`:')
    console.supplement(f'Features shape = {self.features.shape}')
    console.supplement(f'Target shape = {self.targets.shape}')
    console.supplement(f'Groups:')
    for i, group in enumerate(self.groups): console.supplement(
      f'{self.target_labels[i]}: {len(group)} samples', level=2)

    if len(self.target_collection) > 0:
      console.supplement(f'Target keys:')
      for key in self.target_collection.keys():
        console.supplement(f'{key}', level=2)

    if kwargs.get('report_stat', True):
      mu_min, mu_max = min(self.feature_mean[0]), max(self.feature_mean[0])
      console.supplement(f'Feature mean in [{mu_min}, {mu_max}]')

      sigma_min, sigma_max = min(self.feature_std[0]), max(self.feature_std[0])
      console.supplement(f'Feature std in [{sigma_min}, {sigma_max}]')

    # This is for validating split results when shuffle is False
    if kwargs.get('report_feature_sum', False):
      console.supplement(f'Feature sum = {np.sum(self.features)}')

  def standardize(self, omix=None, update_self=False):
    if omix is None: omix = self
    result = self if update_self else self.duplicate()
    mu, sigma = omix.feature_mean, omix.feature_std
    result.features = (omix.features - mu) / (sigma + 1e-6)
    return result

  def duplicate(self, **kwargs) -> 'Omix':
    omix = Omix(
      features=kwargs.get('features', self.features.copy()),
      targets=kwargs.get('targets', self.targets.copy()),
      feature_labels=kwargs.get('feature_labels', self._feature_labels),
      sample_labels=kwargs.get('sample_labels', self.sample_labels),
      target_labels=kwargs.get('target_labels', self._target_labels),
      data_name=kwargs.get('data_name', self.data_name))
    return omix

  def get_k_folds(self, k: int, shuffle=True, random_state=None,
                  balance_classes=True, return_whole=False):
    ratios = [1] * k
    omices = self.split(*ratios, balance_classes=balance_classes,
                        shuffle=shuffle, random_state=random_state,
                        data_labels=[f'Fold-{i + 1} Test' for i in range(k)])

    folds = []
    for i, om_test in enumerate(omices):
      train_omices = omices.copy()
      train_omices.pop(i)
      om_train = Omix.sum(train_omices, data_name=f'Fold-{i + 1} Train')
      folds.append((om_train, om_test))

    if return_whole:
      whole = Omix.sum([o for _, o in folds], data_name='Whole')
      return folds, whole
    return folds

  def split(self, *ratios, data_labels=None, balance_classes=True,
            shuffle=True, random_state=None) -> List['Omix']:
    from sklearn.model_selection import train_test_split

    # Sanity check
    if data_labels is None:
      data_labels = [f'{self.data_name}-{i + 1}' for i in range(len(ratios))]
    assert len(ratios) == len(data_labels), '!! ratios and data_labels must have the same length'
    assert len(ratios) > 1, '!! At least two splits are required'

    X, y, labels = self.features, self.targets, self.sample_labels

    if len(ratios) == 2:
      # End of recursion
      test_size = ratios[1] / sum(ratios)
      stratify = y if balance_classes else None

      if balance_classes and not shuffle:
        # Note that this is not supported by train_test_split
        om1_list, om2_list = [], []
        for om in self.omix_groups:
          om1, om2 = om.split(*ratios, balance_classes=False, shuffle=False)
          om1_list.append(om1), om2_list.append(om2)
        om1: Omix = Omix.sum(om1_list, data_name=data_labels[0])
        om2: Omix = Omix.sum(om2_list, data_name=data_labels[1])
      else:
        # Split data using train_test_split
        X_train, X_test, y_train, y_test, l_train, l_test = train_test_split(
          X, y, labels, test_size=test_size, random_state=random_state,
          shuffle=shuffle, stratify=stratify)

        om1 = self.duplicate(features=X_train, targets=y_train,
                             sample_labels=l_train, data_name=data_labels[0])
        om2 = self.duplicate(features=X_test, targets=y_test,
                             sample_labels=l_test, data_name=data_labels[1])

      return [om1, om2]
    else:
      # Recursively split
      om1, om2 = self.split(
        sum(ratios[:-1]), ratios[-1], data_labels=['temp', data_labels[-1]],
        balance_classes=balance_classes, random_state=random_state,
        shuffle=shuffle)
      return om1.split(*ratios[:-1], data_labels=data_labels[:-1],
                       balance_classes=balance_classes, shuffle=shuffle,
                       random_state=random_state) + [om2]

  def intersect_merge(self, other: 'Omix', data_name=None) -> 'Omix':
    console.show_status(
      f'Merging `{self.data_name}` and `{other.data_name}` ...')
    console.section('Information of two Omices')
    self.report()
    print()
    other.report()

    s_indices, o_indices = [], []
    for i, label in enumerate(self.sample_labels):
      js = np.where(other.sample_labels == label)[0]
      # assert len(js) < 2
      if len(js) == 0: continue
      j = js[0]
      assert self.targets[i] == other.targets[j]
      s_indices.append(i), o_indices.append(j)

    s_features, o_features = self.features[s_indices], other.features[o_indices]
    features = np.concatenate((s_features, o_features), axis=1)
    feature_labels = self.feature_labels + other.feature_labels
    sample_labels = self.sample_labels[s_indices]
    if data_name is None: data_name = f'{self.data_name}x{other.data_name}'

    s_targets, o_targets = self.targets[s_indices], other.targets[o_indices]

    omix = self.duplicate(features=features, feature_labels=feature_labels,
                          targets=self.targets[s_indices],
                          sample_labels=sample_labels, data_name=data_name)

    console.section('Information after merging')
    omix.report()
    return omix

  # endregion: Public Methods

  # region: Overriding

  @staticmethod
  def sum(omices: List['Omix'], data_name='Sum') -> 'Omix':
    om = sum(omices[1:], start=omices[0])
    om.data_name = data_name
    return om

  def __add__(self, other):
    assert isinstance(other, Omix), '!! other must be an instance of Omix'

    data_name = f'{self.data_name} + {other.data_name}'
    return self.duplicate(
      features=np.concatenate((self.features, other.features)),
      targets=np.concatenate((self.targets, other.targets)),
      sample_labels=np.concatenate((self.sample_labels, other.sample_labels)),
      data_name=data_name)

  def __mul__(self, other):
    assert isinstance(other, Omix), '!! other must be an instance of Omix'

    features = np.concatenate((self.features, other.features), axis=1)
    assert all(self.targets == other.targets), '!! targets must be the same'
    assert all(self.sample_labels == other.sample_labels), '!! sample_labels must be the same'

    if self._feature_labels is None and other._feature_labels is None:
      feature_labels = None
    else: feature_labels = self.feature_labels + other.feature_labels

    assert self.target_labels == other.target_labels, '!! target_labels must be the same'

    data_name = f'{self.data_name} x {other.data_name}'
    return self.duplicate(features=features, feature_labels=feature_labels,
                          data_name=data_name)

  def __str__(self):
    return f'{self.data_name}({self.n_samples}x{self.n_features})'

  def __repr__(self): return str(self)

  # endregion: Overriding



if __name__ == '__main__':
  pass
