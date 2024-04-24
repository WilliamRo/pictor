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

  def multi_plots(self, indices: str = '1-10', rows: int = 2):
    """Plot multiple features at once

    Args:
      indices: str, can be
        (1) '1,4,6,2'
        (2) '1-5'
    """
    if '-' in indices:
      start, end = map(int, indices.split('-'))
      indices = list(range(start, end + 1))
    else:
      indices = list(map(int, indices.split(',')))

    N = len(indices)
    cols = (N + 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
    for i, ax in zip(indices, axes.flatten()):
      self.plot(self.pictor.objects[i - 1], ax,
                max_title_len=15, n_top=rows + 1)

    plt.tight_layout()
    plt.show()

  mp = multi_plots


  def cross_plots(self, indices: str = '1-3'):
    """Generate a cross plot

    Args:
      indices: str, can be
        (1) '1,4,6'
        (2) '1-3'
    """
    if '-' in indices:
      start, end = map(int, indices.split('-'))
      indices = list(range(start, end + 1))
    else:
      indices = list(map(int, indices.split(',')))

    N = len(indices)

    # (1) Draw diagonal plots
    diag_axes = []
    for i in range(N):
      ax = plt.subplot(N, N, i * N + i + 1)
      fi = self.pictor.objects[indices[i] - 1]
      self.plot(fi, ax, max_title_len=15, n_top=N + 1)
      diag_axes.append(ax)

    # (2) Draw upper triangle plots
    for i in range(N):
      for j in range(i + 1, N):
        fi, fj = [self.pictor.objects[indices[k] - 1] for k in [i, j]]
        ax: plt.Axes = plt.subplot(N, N, i * N + j + 1)

        x1, x2 = self.features[:, fi], self.features[:, fj]
        groups = [(x1[self.targets == k], x2[self.targets == k])
                  for k, _ in enumerate(self.target_labels)]
        for tid, (_x1, _x2) in enumerate(groups):
          color = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w'][tid]
          ax.scatter(_x2, _x1, alpha=0.3, c=color, s=2,
                     label=self.target_labels[tid])
        ax.set_xlim(diag_axes[j].get_ylim())
        ax.set_ylim(diag_axes[i].get_ylim())
        ax.legend()

    plt.gcf().set_figheight(7)
    plt.gcf().set_figwidth(7)
    plt.tight_layout()
    plt.show()

  cp = cross_plots


  def plot(self, x, ax: plt.Axes, max_title_len=999, **kwargs):
    features = self.features[:, x]
    groups = [features[self.targets == i]
              for i, _ in enumerate(self.target_labels)]

    box = ax.boxplot(groups, showfliers=False,
                     positions=range(len(groups)))
    ax.set_xticklabels(self.target_labels)
    ax.set_title(self.feature_labels[x][:max_title_len])

    # Show statistical annotation if required
    if self.get('statanno'):
      ann = Annotator(groups, ax)
      ann.annotate(**kwargs)

  # endregion: Plot Methods

  # region: Commands

  def sort(self, sort_by='p_val'):
    assert sort_by == 'p_val'
    console.show_status('Sorting by p-values ...')
    with self.pictor.busy('Sorting ...'):
      indices = np.argsort(
        [r[0][2] for r in self.omix.single_factor_analysis_reports])
    self.pictor.objects = indices
    self.refresh()
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
    p.objects = list(range(omix.features.shape[1]))
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
