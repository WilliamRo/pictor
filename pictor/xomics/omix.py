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
from collections import OrderedDict
from collections.abc import Iterable
from roma import console
from roma import io
from roma import Nomear
from typing import List

import numpy as np
import os



class Omix(Nomear):

  class Keys:
    DimensionReducer = 'Omix::DimensionReducer'

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
    from pictor.xomics.stat_analyzers import single_factor_analysis

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
  def dimension_reducer(self):
    return self.get_from_pocket(self.Keys.DimensionReducer)

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

  def select_features(self, method: str, shadow_omix=None, **kwargs):
    from pictor.xomics.ml.dr import get_reducer_class, DREngine

    # (0) Get configs
    save_model = kwargs.get('save_model', False)

    if shadow_omix is not None:
      # This branch is activated when the shadow_omix is provided,
      # currently only used in Pipeline.create_sub_space to avoid repeatedly
      # create deterministic sub-spaces generated by method such as PCA
      reducer = shadow_omix.dimension_reducer
      omix_reduced = shadow_omix.duplicate()
    else:
      # (1) Reduce dimension, return reduced omix
      ReducerClass = get_reducer_class(method)

      reducer: DREngine = ReducerClass(**kwargs)
      reducer.fit_reducer(self, **kwargs)
      omix_reduced = reducer.reduce_dimension(self, **kwargs)

    if save_model: omix_reduced.put_into_pocket(
      self.Keys.DimensionReducer, reducer, local=True)

    return omix_reduced

  def get_sub_space(self, indices, start_from_1=True) -> 'Omix':
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

  def filter_by_name(self, keywords: List[str], include=True):
    """Filter features by keywords in feature_labels"""
    if not isinstance(keywords, list): keywords = [keywords]

    indices = []
    for i, label in enumerate(self.feature_labels):
      matched = any([key.lower() in label.lower() for key in keywords])
      if matched and include: indices.append(i)
      elif not matched and not include: indices.append(i)
    return self.get_sub_space(indices, start_from_1=False)

  # endregion: Feature Selection

  # region: Pipeline

  def pipeline(self, sf: str = 'ucp', ml: str = 'lr',
               r: int = 1, k: int = 50, t: float = 0.8, report: int = 0):
    """Run `feature selection` -> `machine learning` pipeline

    :param sf: feature selection methods. Examples: 'ucp;lasso', 'lasso'
    :param ml: machine learning methods. Examples: 'lr;svm', 'lr'
    :param r: repeat times
    :param k: number of features to select (if applicable)
    :param t: threshold for feature selection (if applicable)
    :param report: 0 for no report, 1 for report, 2 for report and plot
    """
    from pictor.xomics.evaluation.pipeline import Pipeline

    pi = Pipeline(self, ignore_warnings=1, save_models=0)

    # (1) Feature selection
    kwargs = {'repeats': r, 'nested': True, 'threshold': t,
              'k': k, 'show_progress': True}
    for method in sf.split(';'): pi.create_sub_space(method, **kwargs)

    # (2) Machine learning
    kwargs = {'repeats': r, 'nested': True, 'show_progress': True,
              'n_splits': 2, 'verbose': True}
    for method in ml.split(';'):
      pi.fit_traverse_spaces(method, **kwargs)

    # (3) Report
    if report > 0: pi.report()
    if report > 1: pi.plot_matrix()

    return pi

  # endregion: Pipeline

  # region: Visualization

  def show_in_explorer(self, title=None, fig_size=(5, 5), ignore_warnings=True):
    from pictor.xomics import FeatureExplorer

    if title is None: title = self.data_name
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

      # (1.1) Find legal indices from the first row
      indices, illegal_indices = [], []
      for i in range(features.shape[1]):
        try:
          # (1.1.1) Exclude non-numerical values
          np.float32(features[0, i])
          # (1.1.2) Exclude columns with NaN
          x = np.array(features[:, i], dtype=np.float32)
          if np.isnan(x).any(): raise ValueError
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

    nan_mask = np.isnan(targets)

    if nan_mask.any():
      assert return_new_omix, '!! nan_mask is not supported when return_new_omix is False'
      console.show_status(f'Sample number reduced from {len(self.sample_labels)} '
                          f'to {sum(~nan_mask)}.')

    if return_new_omix:
      mask = ~nan_mask
      features = self.features[mask]
      targets = targets[mask]
      sample_labels = np.array(self.sample_labels)[mask]

      return self.duplicate(features=features, targets=targets,
                            sample_labels=sample_labels,
                            target_labels=target_labels)

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

    if not self.targets_are_numerical:
      console.supplement(f'Groups:')
      for i, group in enumerate(self.groups): console.supplement(
        f'{self.target_labels[i]}: {len(group)} samples', level=2)
    else:
      min_target, max_target = min(self.targets), max(self.targets)
      console.supplement(f'Target range: {min_target} - {max_target}')

    if len(self.target_collection) > 0:
      console.supplement(f'Target keys:')
      for key in self.target_collection.keys():
        console.supplement(f'{key}', level=2)

    if kwargs.get('report_stat', False):
      mu_min, mu_max = min(self.feature_mean[0]), max(self.feature_mean[0])
      console.supplement(f'Feature mean in [{mu_min}, {mu_max}]')

      sigma_min, sigma_max = min(self.feature_std[0]), max(self.feature_std[0])
      console.supplement(f'Feature std in [{sigma_min}, {sigma_max}]')

    # This is for validating split results when shuffle is False
    if kwargs.get('report_feature_sum', False):
      console.supplement(f'Feature sum = {np.sum(self.features)}')

  def standardize(self, omix=None, update_self=False, return_mu_sigma=False,
                  mu=None, sigma=None):
    if omix is None: omix = self
    result = self if update_self else self.duplicate()

    if mu is None and sigma is None:
      mu, sigma = omix.feature_mean, omix.feature_std

    result.features = (omix.features - mu) / (sigma + 1e-6)
    if return_mu_sigma: return result, mu, sigma
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
                  balance_classes=True, return_whole=False, **kwargs):
    ratios = [1] * k
    if self.targets_are_numerical: balance_classes = False
    omices = self.split(*ratios, balance_classes=balance_classes,
                        shuffle=shuffle, random_state=random_state,
                        data_labels=[f'Fold-{i + 1} Test' for i in range(k)])

    folds = []
    for i, om_test in enumerate(omices):
      train_omices = omices.copy()
      train_omices.pop(i)
      om_train = Omix.sum(train_omices, data_name=f'Fold-{i + 1} Train')
      folds.append((om_train, om_test))

    # Developer's note: This is for debugging
    if 'sanity_check' in kwargs or True:
      test_omix = Omix.sum([o for _, o in folds], data_name='Test-Omix')
      assert all([sl1 == sl2 for sl1, sl2 in zip(
        sorted(self.sample_labels), sorted(test_omix.sample_labels))])

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
    """Merge features of common samples in two Omices"""
    console.show_status(
      f'Merging `{self.data_name}` and `{other.data_name}` ...')
    console.section('Information of two Omices')
    self.report()
    print()
    other.report()

    s_indices, o_indices = [], []
    for i, label in enumerate(self.sample_labels):
      js = np.where(np.ndarray(other.sample_labels) == label)[0]
      # assert len(js) < 2
      if len(js) == 0: continue
      j = js[0]
      assert self.targets[i] == other.targets[j]
      s_indices.append(i), o_indices.append(j)

    s_features, o_features = self.features[s_indices], other.features[o_indices]
    features = np.concatenate((s_features, o_features), axis=1)
    feature_labels = list(self.feature_labels) + list(other.feature_labels)
    sample_labels = self.sample_labels[s_indices]
    if data_name is None: data_name = f'{self.data_name} x {other.data_name}'

    s_targets, o_targets = self.targets[s_indices], other.targets[o_indices]

    omix = self.duplicate(features=features, feature_labels=feature_labels,
                          targets=self.targets[s_indices],
                          sample_labels=sample_labels, data_name=data_name)

    console.section('Information after merging')
    omix.report()
    return omix

  def select_samples(self, indices: Iterable=None, data_name='Selected',
                     target_labels=None) -> 'Omix':
    dup_dict = {}

    if target_labels is not None:
      indices = []
      targets = []
      for i, tl in enumerate(target_labels):
        group_index = self.target_labels.index(tl)
        indices.extend(self.groups[group_index])
        targets.extend([i] * len(self.groups[group_index]))

      # Target labels should be reset
      dup_dict['target_labels'] = target_labels
      dup_dict['targets'] = np.array(targets)

    assert isinstance(indices, Iterable)

    label_array = np.array(self.sample_labels)
    features, targets, sample_labels = [], [], []
    for i in indices:
      if not isinstance(i, np.integer):
        assert i in self.sample_labels, f'!! {i} must be in sample_labels'
        i = np.where(label_array == i)[0][0]

      features.append(self.features[i])
      targets.append(self.targets[i])
      sample_labels.append(self.sample_labels[i])

    features = np.stack(features, axis=0)

    if 'targets' not in dup_dict: dup_dict['targets'] = targets
    return self.duplicate(features=features, sample_labels=sample_labels,
                          data_name=data_name, **dup_dict)

  @staticmethod
  def gen_psudo_omix(n_samples: int, n_features: int, targets=None,
                     binary_column_indices=None) -> 'Omix':
    """Generate a pseudo Omix with random features and targets"""
    features = np.random.rand(n_samples, n_features)

    # Set binary features if specified
    if binary_column_indices is not None:
      for i in binary_column_indices:
        features[:, i] = np.random.randint(0, 2, size=n_samples)

    if targets is None: targets = np.random.randint(0, 2, size=n_samples)  # Binary targets
    return Omix(features, targets,
                target_labels=('Negative', 'Positive'), data_name='Pseudo Omix')

  # endregion: Public Methods

  # region: Overriding

  @staticmethod
  def sum(omices: List['Omix'], data_name='Sum') -> 'Omix':
    om = sum(omices[1:], start=omices[0])
    om.data_name = data_name
    return om

  def __add__(self, other):
    """Merge samples"""
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
    ssl, osl = [np.array(om.sample_labels) for om in (self, other)]
    assert all(ssl == osl), '!! sample_labels must be the same'

    if self._feature_labels is None and other._feature_labels is None:
      feature_labels = None
    else: feature_labels = np.concatenate(
      [self.feature_labels, other.feature_labels])

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
