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
from roma import io
from roma import Nomear

import numpy as np



class Omix(Nomear):

  def __init__(self, features, targets, feature_labels=None,
               target_labels=None, data_name='Omix'):
    self.features = features
    self.targets = targets

    # Sanity check
    assert len(features) == len(targets), '!! features and targets must have the same length'
    assert len(features.shape) == 2, '!! features must be a 2D array'

    self._feature_labels = feature_labels
    self._target_labels = target_labels
    self.data_name = data_name


  # region: Properties

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

  # endregion: Properties

  # region: Feature Selection



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


  # endregion: Public Methods

  # region: Overriding

  def __add__(self, other):
    assert isinstance(other, Omix), '!! other must be an instance of Omix'

    features = np.concatenate((self.features, other.features), axis=1)
    assert all(self.targets == other.targets), '!! targets must be the same'

    if self._feature_labels is None and other._feature_labels is None:
      feature_labels = None
    else: feature_labels = self.feature_labels + other.feature_labels

    assert self.target_labels == other.target_labels, '!! target_labels must be the same'

    data_name = f'{self.data_name} + {other.data_name}'
    return Omix(features, self.targets, feature_labels, self.target_labels, data_name)

  # endregion: Overriding



if __name__ == '__main__':
  pass
