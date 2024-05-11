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
from pictor.xomics.stat_analyzers import single_factor_analysis
from roma import console
from roma import io
from roma import Nomear
from typing import List

import numpy as np



class Omix(Nomear):

  def __init__(self, features, targets, feature_labels=None,
               target_labels=None, data_name='Omix'):
    """features.shape = [n_samples, n_features]"""
    self.features = features
    self.targets = targets

    # Sanity check
    assert len(features) == len(targets), '!! features and targets must have the same length'
    assert len(features.shape) == 2, '!! features must be a 2D array'

    self._feature_labels = feature_labels
    self._target_labels = target_labels
    self.data_name = data_name


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

  @Nomear.property()
  def target_labels(self):
    if self._target_labels is None:
      return [f'class-{i + 1}' for i in range(len(np.unique(self.targets)))]
    return self._target_labels

  @Nomear.property()
  def groups(self):
    """Lists of indices of samples in each class"""
    return [np.where(self.targets == i)[0] for i in np.unique(self.targets)]

  @Nomear.property()
  def omix_groups(self):
    """Returns omix objects for each group"""
    return [self.duplicate(features=self.features[group],
                           targets=self.targets[group])
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
  def feature_mean(self): return np.mean(self.features, axis=0, keepdims=True)

  @Nomear.property()
  def feature_std(self): return np.std(self.features, axis=0, keepdims=True)

  @Nomear.property()
  def corr_matrix(self): return np.corrcoef(self.features, rowvar=False)

  # endregion: Properties

  # region: Feature Selection

  def select_features(self, method: str, **kwargs):
    method = method.lower()

    omix = self.standardize() if kwargs.get('standardize', 1) else self

    if method in ('pca', ):
      from sklearn.decomposition import PCA

      n_components = kwargs.get('n_components', 3)

      pca = PCA(n_components=n_components)
      feature_labels = [f'PC-{i + 1}' for i in range(n_components)]
      omix_reduced = self.duplicate(features=pca.fit_transform(omix.features),
                                    feature_labels=feature_labels)
    elif method in ('lasso', ):
      from pictor.xomics.ml.lasso import Lasso
      omix_reduced = Lasso(
        kwargs.get('verbose', 0)).select_features(omix, **kwargs)
    else: raise KeyError(f'!! Unknown feature selecting method "{method}"')

    return omix_reduced

  # endregion: Feature Selection

  # region: Visualization

  def show_in_explorer(self, title='Omix'):
    from pictor.xomics import FeatureExplorer
    FeatureExplorer.explore(omix=self, title=title)

  # endregion: Visualization

  # region: IO

  @staticmethod
  def load(file_path: str, verbose=True):
    return io.load_file(file_path, verbose=verbose)

  def save(self, file_path: str, verbose=True):
    if not file_path.endswith('.omix'): file_path += '.omix'
    return io.save_file(self, file_path, verbose=verbose)

  # endregion: IO

  # region: Public Methods

  def report(self, **kwargs):
    console.show_info(f'Details of `{self.data_name}`:')
    console.supplement(f'Features shape = {self.features.shape}')
    console.supplement(f'Target shape = {self.targets.shape}')
    console.supplement(f'Groups:')
    for i, group in enumerate(self.groups): console.supplement(
      f'{self.target_labels[i]}: {len(group)} samples', level=2)

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
    result.features = (omix.features - mu) / sigma
    return result

  def duplicate(self, **kwargs) -> 'Omix':
    features = kwargs.get('features', self.features.copy())
    targets = kwargs.get('targets', self.targets.copy())
    feature_labels = kwargs.get('feature_labels', self._feature_labels)
    target_labels = kwargs.get('target_labels', self._target_labels)
    data_name = kwargs.get('data_name', self.data_name)
    return Omix(features, targets, feature_labels, target_labels, data_name)

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

    X, y = self.features, self.targets

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
        X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=test_size, random_state=random_state,
          shuffle=shuffle, stratify=stratify)

        om1 = self.duplicate(
          features=X_train, targets=y_train, data_name=data_labels[0])
        om2 = self.duplicate(
          features=X_test, targets=y_test, data_name=data_labels[1])

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
      data_name=data_name)

  def __mul__(self, other):
    assert isinstance(other, Omix), '!! other must be an instance of Omix'

    features = np.concatenate((self.features, other.features), axis=1)
    assert all(self.targets == other.targets), '!! targets must be the same'

    if self._feature_labels is None and other._feature_labels is None:
      feature_labels = None
    else: feature_labels = self.feature_labels + other.feature_labels

    assert self.target_labels == other.target_labels, '!! target_labels must be the same'

    data_name = f'{self.data_name} x {other.data_name}'
    return Omix(features, self.targets, feature_labels, self.target_labels,
                data_name)

  def __str__(self):
    return f'{self.data_name}({self.n_samples}x{self.n_features})'

  def __repr__(self): return str(self)

  # endregion: Overriding



if __name__ == '__main__':
  pass