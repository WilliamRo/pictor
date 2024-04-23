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
# ====-======================================================================-==
from pictor.xomics.stat_annotator import Annotator
from pictor.xomics.stat_analyzers import single_factor_analysis
from pictor import Plotter
from roma import check_type
from roma import console

import matplotlib.pyplot as plt
import numpy as np



class FeatureExplorer(Plotter):

  class Keys:
    features = 'features'
    feature_labels = 'feature_labels'
    targets = 'targets'
    target_labels = 'target_labels'

  def __init__(self):
    super(FeatureExplorer, self).__init__(self.plot)

    self.new_settable_attr('statanno', True, bool, 'Statistical annotation')

  # region: Properties

  @Plotter.property()
  def features(self): raise ValueError('!! features not found')

  @Plotter.property()
  def feature_labels(self): raise ValueError('!! feature_labels not found')

  @Plotter.property()
  def targets(self): raise ValueError('!! targets not found')

  @Plotter.property()
  def target_labels(self): raise ValueError('!! target_labels not found')

  @property
  def num_classes(self): return len(self.target_labels)

  @Plotter.property()
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

  # region: Plot Methods

  def plot(self, x, ax: plt.Axes):
    features = self.features[:, x]
    groups = [features[self.targets == i]
              for i, _ in enumerate(self.target_labels)]

    box = ax.boxplot(groups, showfliers=False,
                     positions=range(len(groups)))
    ax.set_xticklabels(self.target_labels)
    ax.set_title(self.feature_labels[x])

    # Show statistical annotation if required
    if self.get('statanno'):
      ann = Annotator(groups, ax)
      ann.annotate()

  # endregion: Plot Methods

  # region: Commands

  def sort(self, sort_by='p_val'):
    assert sort_by == 'p_val'
    console.show_status('Sorting by p-values ...')
    with self.pictor.busy('Sorting ...'):
      indices = np.argsort(
        [r[0][2] for r in self.single_factor_analysis_reports])
    self.pictor.objects = indices
    return indices

  def register_shortcuts(self):
    self.register_a_shortcut(
      'a', lambda: self.flip('statanno'), 'Toggle statanno')

  # endregion: Commands

  # region: Public Methods

  @classmethod
  def explore(cls, features, targets, feature_labels=None, target_labels=None,
              title='Feature Explorer', fig_size=(5, 5), auto_show=True):
    from pictor import Pictor

    # 1. Sanity check
    features: np.ndarray = check_type(features, np.ndarray)
    targets: np.ndarray = check_type(targets, np.ndarray)
    assert len(features.shape) == 2 and len(targets.shape) == 1
    assert features.shape[0] == targets.shape[0]

    if feature_labels is None:
      feature_labels = [f'feature-{i+1}' for i in range(features.shape[1])]
    if target_labels is None:
      target_labels = [f'target-{i+1}' for i in range(len(set(targets)))]

    # 2. Initiate Pictor and show
    p = Pictor(title=title, figure_size=fig_size)
    p.objects = list(range(features.shape[1]))
    p.labels = feature_labels

    fe = cls()
    fe.put_into_pocket(fe.Keys.features, features)
    fe.put_into_pocket(fe.Keys.feature_labels, feature_labels)
    fe.put_into_pocket(fe.Keys.targets, targets)
    fe.put_into_pocket(fe.Keys.target_labels, target_labels)
    p.add_plotter(fe)

    if auto_show: p.show()
    return p, fe

  # endregion: Public Methods



if __name__ == '__main__':
  from sklearn import datasets

  iris = datasets.load_iris()

  FeatureExplorer.explore(
    iris.data, iris.target, feature_labels=iris.feature_names,
    target_labels=iris.target_names, title='Iris Explorer')
