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
from pictor.xomics.omix import Omix
from pictor import Plotter
from roma import console

import matplotlib.pyplot as plt
import numpy as np



class FeatureExplorer(Plotter):

  class Keys:
    features = 'features'
    feature_labels = 'feature_labels'
    targets = 'targets'
    target_labels = 'target_labels'

  def __init__(self, omix: Omix):
    super(FeatureExplorer, self).__init__(self.plot)

    self.omix = omix

    self.new_settable_attr('statanno', True, bool, 'Statistical annotation')

  # region: Properties

  @property
  def features(self): return self.omix.features

  @property
  def feature_labels(self): return self.omix.feature_labels

  @property
  def targets(self): return self.omix.targets

  @property
  def target_labels(self): return self.omix.target_labels

  @property
  def num_classes(self): return len(self.target_labels)

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
        [r[0][2] for r in self.omix.single_factor_analysis_reports])
    self.pictor.objects = indices
    return indices

  def register_shortcuts(self):
    self.register_a_shortcut(
      'a', lambda: self.flip('statanno'), 'Toggle statanno')

  # endregion: Commands

  # region: Public Methods

  @classmethod
  def explore(cls, features=None, targets=None, feature_labels=None,
              target_labels=None, omix=None, title='Feature Explorer',
              fig_size=(5, 5), auto_show=True):
    from pictor import Pictor

    # 1. Wrap data
    if omix is None:
      omix = Omix(features, targets, feature_labels, target_labels)

    # 2. Initiate Pictor and show
    p = Pictor(title=title, figure_size=fig_size)
    p.objects = list(range(features.shape[1]))
    p.labels = omix.feature_labels

    fe = cls(omix)
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
