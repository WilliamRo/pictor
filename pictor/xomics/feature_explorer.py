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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pictor.xomics.stat_annotator import Annotator
from pictor.xomics.omix import Omix
from pictor import Plotter
from roma import console

import matplotlib.pyplot as plt
import numpy as np
import os
import warnings



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
    self.new_settable_attr('showfliers', False, bool, 'Option to show fliers')
    self.new_settable_attr('roc', False, bool, 'Option to show ROC curve')

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

  def multi_plots(self, indices: str = '1-10', rows: int = 2,
                  hsize: int = 10, vsize: int = 6):
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
    cols = (N + 1) // rows if rows > 1 else N
    fig, axes = plt.subplots(rows, cols, figsize=(hsize, vsize))
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

  def correlation_plot(self, show_name: int=1, cmap: str='RdBu'):
    """Plot correlation matrix"""
    # Calculate correlation matrix (consider sorted indices)
    indices = np.array(self.pictor.objects)
    matrix = np.corrcoef(self.omix.features[:, indices], rowvar=False)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)

    # Set ticks
    if show_name:
      ticks = np.arange(len(self.feature_labels))
      feature_labels = [self.feature_labels[i] for i in indices]
      ax.set_xticks(ticks=ticks, labels=feature_labels)
      fig.autofmt_xdate(rotation=45)
      ax.set_yticks(ticks=ticks, labels=feature_labels)
    else:
      ax.set_xticks([])
      ax.set_yticks([])

    # Show color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

    # Show
    fig.tight_layout()
    plt.show()

  cop = correlation_plot

  def plot(self, x, ax: plt.Axes, max_title_len=999, **kwargs):
    features = self.features[:, x]

    groups = [features[self.targets == i]
              for i, _ in enumerate(self.target_labels)]

    # (Option-1) Plot ROC if required
    if self.get('roc') and len(groups) == 2:
      from pictor.xomics.evaluation.roc import ROC
      if np.mean(groups[0]) > np.mean(groups[1]): features = -features
      roc = ROC(features, self.targets)
      roc.plot_roc(ax)
      return

    # (Option-2) Plot boxplot
    if len(features) == 1:
      target = self.targets[0]
      ax.plot(target, features[0], 'o')
      ax.set_xticks([0, 1], self.target_labels)
      ax.set_xlim(-0.5, 1.5)
    else:
      ax.boxplot(groups, showfliers=self.get('showfliers'),
                 positions=range(len(groups)))
      ax.set_xticklabels(self.target_labels)

    ax.set_title(self.feature_labels[x][:max_title_len])

    # Show statistical annotation if required
    if self.get('statanno') and all([len(g) > 1 for g in groups]):
      ann = Annotator(groups, ax)
      ann.annotate(**kwargs)

  # endregion: Plot Methods

  # region: Commands

  # region: Feature Selection

  def sf_pca(self, n_components: int=10, standardize: int=1):
    """Feature selection using PCA"""
    self.select_features('PCA', n_components=n_components,
                         standardize=standardize)

  def sf_lasso(self, verbose: int=0, plot_path: int=0, n_splits: int=5,
               strategy: str='grid', random_state: int=None,
               threshold: float=0.001, min_alpha_exp: int=-7,
               max_alpha_exp: int=1, n_alphas: int=100, standardize: int=1,
               n_jobs:int =10, save_model: int=0):
    """Feature selection using Lasso regression.

    - Super Signature Test: Pass
    """
    hp_space = {'alpha': np.logspace(min_alpha_exp, max_alpha_exp, n_alphas)}
    self.select_features(
      'Lasso', n_splits=n_splits, strategy=strategy, hp_space=hp_space,
      random_state=random_state, threshold=threshold, verbose=verbose,
      standardize=standardize, n_jobs=n_jobs, plot_path=plot_path,
      save_model=save_model)

  def sf_mrmr(self, k=10, standardize=1):
    """Feature selection using mRMR.

    - Super Signature Test: Fail
    """
    self.select_features('mRMR', k=k, standardize=standardize)

  def select_features(self, method: str, **kwargs):
    """Select features using a specific method"""
    # key = (method, tuple(kwargs.items()))

    with self.pictor.busy('Selecting features ...'):
      omix = self.omix.select_features(method, **kwargs)

    title = f'{self.pictor.static_title} - {method}'
    omix.show_in_explorer(title=title)

  def switch_target(self, target_key: str, new_omix: int = 1):
    """Switch to a new target"""
    omix = self.omix.set_targets(target_key, return_new_omix=new_omix)
    if new_omix: omix.show_in_explorer()
    else: self.refresh()
  st = switch_target

  # endregion: Feature Selection

  # region: Machine Learning

  def ml(self, model, verbose: int = 1, warning: int = 1, print_cm: int = 0,
         plot_roc: int = 0, plot_cm: int = 0, cm: int = 1, auc: int = 1,
         mi: int = 0, seed: int = None, sig: int = 0, lc: int = 0):
    """Below are the machine learning methods you can use in FeatureExplorer

    Args:
      model: str, model name
        - lr: Logistic Regression Classifier
        - svm: Support Vector Classifier
        - dt: Decision Tree Classifier
        - rf: Random Forest Classifier
        - xgb: XGBoost Classifier

      verbose: int, 0: show fitting status, 1: show fitting details
      cm: int, 1: show confusion matrix
      print_cm: int, 1: print confusion matrix
      plot_cm: int, 1: plot confusion matrix
      auc: int, 1: show AUC
      plot_roc: int, 1: plot ROC curve
      warning: int, 0: ignore warnings
      mi: int, 0: show misclassified sample indices
      seed: int, random seed
      sig: int, 0: option to show signature
      lc: int, 0: option to show learning curve
    """
    from pictor.xomics.ml import get_model_class

    ModelClass = get_model_class(model)

    model = ModelClass(ignore_warnings=warning == 0)
    if lc:
      model.plot_learning_curve(self.omix, verbose=verbose)
      return

    model.fit_k_fold(self.omix, verbose=verbose, cm=cm, print_cm=print_cm,
                     auc=auc, plot_roc=plot_roc, plot_cm=plot_cm, mi=mi,
                     random_state=seed, show_signature=sig == 1)

  # endregion: Machine Learning

  def report(self, report_feature_sum: int=0, report_stat: int=1):
    self.omix.report(report_feature_sum=report_feature_sum,
                     report_stat=report_stat > 0)

  def standardize(self, update_self: int=0):
    """Standardize features"""
    result = self.omix.standardize(update_self=update_self)
    title = f'{self.pictor.static_title} - Standardized'
    if update_self:
      self.pictor.static_title = title
      self.refresh()
    else: result.show_in_explorer(title=title)
  sdd = standardize

  def sort(self, sort_by='p_val'):
    assert sort_by == 'p_val'
    console.show_status('Sorting by p-values ...')
    with self.pictor.busy('Sorting ...'):
      indices = np.argsort(
        [r[0][2] for r in self.omix.single_factor_analysis_reports])
    sorted_omix = self.omix.get_sub_space(indices, start_from_1=False)
    self.refresh()
    sorted_omix.show_in_explorer(title=f'{self.pictor.static_title} - Sorted')
    return indices

  def find(self, key: str):
    """Find next feature by name"""
    current_index = self.pictor.cursors[self.pictor.Keys.OBJECTS]

    labels = [s.lower() for s in self.omix.feature_labels]
    labels = labels[current_index + 1:] + labels[:current_index + 1]

    key = key.lower()
    for i in range(len(labels)):
      if key in labels[i]:
        self.pictor.set_object_cursor((i + 2 + current_index) % len(labels))
        self.refresh()
        return
    console.show_status(f'No feature found with key `{key}`.')

  def filter_by_name(self, key: str):
    """Filter features by their name"""
    keys = key.split(',')
    omix = self.omix.filter_by_name(keys)
    omix.show_in_explorer()
  fbn = filter_by_name

  def register_shortcuts(self):
    self.register_a_shortcut(
      'a', lambda: self.flip('statanno'), 'Toggle statanno')
    self.register_a_shortcut(
      'f', lambda: self.flip('showfliers'), 'Toggle showfliers')
    self.register_a_shortcut(
      'r', lambda: self.flip('roc'), 'Toggle ROC')

  def ls(self):
    """Below are misc methods you can use in FeatureExplorer

    - cp: cross_plots, plot cross plots
    - cop: correlation_plot, plot correlation matrix
    - fbn: filter_by_name, filter features by name
    - find: find next feature by name
    - mp: multi_plots, plot multiple features at one figure
    - report: report omix details
    - sdd: standardize features
    - sort: sort features by p-values
    - st: switch_target, switch to a new target

    - sf: list feature selection methods
    - ml: list machine learning methods
    """
    pass

  def sf(self, indices: str):
    """Below are the feature selection methods you can use in FeatureExplorer

    - sf_lasso: select features using Lasso regression
    - sf_pca: select features using PCA
    - sf_mrmr: select features using mRMR
    """
    self.select_features('indices', indices=indices)

  # endregion: Commands

  # region: Public Methods

  @classmethod
  def explore(cls, features=None, targets=None, feature_labels=None,
              target_labels=None, omix=None, title='Feature Explorer',
              fig_size=(5, 5), auto_show=True, ignore_warnings=False):
    from pictor import Pictor

    # 0. Ignore warnings if required
    if ignore_warnings:
      warnings.simplefilter('ignore')
      os.environ["PYTHONWARNINGS"] = "ignore"
      console.show_status('Warning Ignored.', prompt='[FE] >>')

    # 1. Wrap data
    if omix is None:
      omix = Omix(features, targets, sample_labels=None,
                  feature_labels=feature_labels,
                  target_labels=target_labels)

    # 2. Initiate Pictor and show
    p = Pictor(title=title, figure_size=fig_size)
    p.objects = list(range(omix.features.shape[1]))
    p.labels = omix.feature_labels

    fe = cls(omix)
    p.add_plotter(fe)

    if auto_show: p.show()
    return p, fe

  def save(self):
    """Save the current omix object"""
    import tkinter as tk

    file_path = tk.filedialog.asksaveasfilename(
      title='Save as', filetypes=[('OMIX files', '*.omix')])

    if file_path is not None: self.omix.save(file_path, verbose=True)

  # endregion: Public Methods



if __name__ == '__main__':
  from sklearn import datasets

  iris = datasets.load_iris()

  auto_show = 0
  p, fe = FeatureExplorer.explore(
    iris.data, iris.target, feature_labels=iris.feature_names,
    target_labels=iris.target_names, title='Iris Explorer', auto_show=auto_show)

  p.show()

  # o = fe.omix
  # o.report()
  # print()
  #
  # shuffle = 1
  # o1, o2 = o.split(4, 1, data_labels=['Data_1', 'Data_2'], shuffle=shuffle == 1)
  # o1.report(report_feature_sum=True), o2.report(report_feature_sum=True)


