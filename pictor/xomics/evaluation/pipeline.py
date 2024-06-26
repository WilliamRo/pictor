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
# ====-====================================================================-====
from collections import OrderedDict
from pictor.xomics.ml.dr.dr_engine import DREngine
from pictor.xomics.ml.ml_engine import FitPackage
from pictor.xomics.omix import Omix
from pictor.xomics.stat_analyzers import calc_CI
from roma import console
from roma import Nomear

import os
import warnings
import numpy as np



class Pipeline(Nomear):
  """Omix-based pipeline for feature selection, fitting, and evaluation.
  """
  prompt = '[PIPELINE] >>'

  def __init__(self, omix: Omix, ignore_warnings=False, save_models=False):
    # 0. Ignore warnings if required
    if ignore_warnings:
      warnings.simplefilter('ignore')
      os.environ["PYTHONWARNINGS"] = "ignore"
      console.show_status('Warning Ignored.', prompt=self.prompt)

    self.omix: Omix = omix
    self.save_models = save_models

  # region: Properties

  @property
  def sub_space_dict(self) -> OrderedDict:
    """Format: {('method_key', (('arg1', val1), ('arg2', val2), ...)):
       [omix_1, omix_2, ...], ...}"""
    return self.omix.get_from_pocket(
      'pp::sub_space_dict::24ma14',
      initializer=lambda: OrderedDict(), local=True)

  @property
  def sub_spaces(self) -> list:
    spaces = []
    for _, omices in self.sub_space_dict.items(): spaces.extend(omices)
    return spaces

  @property
  def lasso_dim_median(self):
    dims = []
    for key, omices in self.sub_space_dict.items():
      if 'lasso' not in key[0].lower(): continue
      dims.extend([omix.n_features for omix in omices])
    return int(np.median(dims))

  @property
  def pipeline_ranking(self):
    """[..., AUC, DR_Model, ML_PKG), ...]"""
    ranking = []
    for _, omix_list in self.sub_space_dict.items():
      # Gather dimension reducers, in case any omix is duplicated thus has no dr
      # ... even save_model is True
      reducers = [omix.dimension_reducer
                 for omix in omix_list if omix.dimension_reducer is not None]
      shared_reducer = reducers[0] if len(reducers) == 1 else None

      for omix in omix_list:
        assert isinstance(omix, Omix)
        pkg_dict: OrderedDict = self.get_fit_packages(omix)
        for _, pkg_list in pkg_dict.items():
          for pkg in pkg_list:
            reducer = omix.dimension_reducer
            if reducer is None: reducer = shared_reducer
            ranking.append((pkg['AUC'], reducer, pkg))

    return sorted(ranking, key=lambda x: x[0], reverse=True)

  # endregion: Properties

  # region: Feature Selection

  def create_sub_space(self, method: str='*', repeats=1,
                       show_progress=0, **kwargs):
    method = method.lower()
    prompt = '[FEATURE SELECTION] >>'

    # if method == 'pca': assert repeats == 1, "Repeat PCA makes no sense."
    if 'save_model' not in kwargs: kwargs['save_model'] = self.save_models

    # Initialize bag if not exists
    key = (method, tuple(sorted(tuple(kwargs.items()), key=lambda x: x[0])))
    if key not in self.sub_space_dict: self.sub_space_dict[key] = []

    if show_progress: console.show_status(
      f'Creating sub-feature space using `{method}` ...', prompt=prompt)

    for i in range(repeats):
      if show_progress: console.print_progress(i, repeats)
      if method == '*': omix_sub = self.omix
      else:
        if method == 'pca' and i > 0: omix_sub = omix_sub.duplicate()
        else: omix_sub = self.omix.select_features(method, **kwargs)

      self.sub_space_dict[key].append(omix_sub)

    if show_progress: console.show_status(
      f'{repeats} sub-feature spaces created.', prompt=prompt)

  # endregion: Feature Selection

  # region: Fitting

  def fit_traverse_spaces(self, model: str, repeats=1, nested=True,
                          show_progress=0, **kwargs):
    from pictor.xomics.ml import get_model_class
    from pictor.xomics.ml.ml_engine import MLEngine

    # (0) Get settings
    prompt = '[PP_FIT] >>'
    verbose = kwargs.get('verbose', 0)

    # (1) Initiate a model
    ModelClass = get_model_class(model)
    model: MLEngine = ModelClass()

    # (2) Traverse
    sub_spaces = self.sub_spaces
    N = len(sub_spaces)
    nested_suffix = ', nested' if nested else ''
    if show_progress: console.show_status(
      f'Traverse through {N} subspaces (repeat={repeats}{nested_suffix}) using {model} ...',
      prompt=prompt)

    for i, omix in enumerate(sub_spaces):
      if show_progress: console.print_progress(i, N)

      pkg_dict = self.get_fit_packages(omix)
      model_name = str(model)
      if model_name not in pkg_dict: pkg_dict[model_name] = []

      # (2.1) tune hyper-parameters
      if not nested: hp = model.tune_hyperparameters(omix, verbose=verbose)
      else: hp = None

      # (2.2) Repeatedly fit model on omix
      for _ in range(repeats):
        pkg = model.fit_k_fold(omix, hp=hp, save_models=self.save_models,
                               nested=nested, **kwargs)
        pkg_dict[model_name].append(pkg)

    if show_progress: console.show_status(
      f'Traversed through {N} subspaces for {repeats} times.', prompt=prompt)

  # endregion: Fitting

  # region: Public Methods

  def get_fit_packages(self, omix: Omix) -> OrderedDict:
    """Format: {'model_name': [pkg_1, pkg_2, ...], ...}"""
    return omix.get_from_pocket(
      'pp::fit_packages::24ma14', initializer=lambda: OrderedDict(), local=True)

  def get_pkg_matrix(self, abbreviate=False):
    from pictor.xomics.ml import abbreviation_dict

    row_labels, col_labels, matrix_dict = [], [], {}

    # For each sub-space
    for key, omix_list in self.sub_space_dict.items():
      # key = ('sf_method_name', (('arg_1', arg_1_value), ...))
      sf_key = key[0]

      # Add feature number to specific methods
      if sf_key in ('pca', 'mrmr', 'sig', 'pval'):
        arg, val = key[1][0]
        if arg in ('n_components', 'k'):
          sf_key += f'-{val}'

      # Register sf_key if not exists
      if sf_key not in row_labels: row_labels.append(sf_key)

      for omix in omix_list:
        # {'model_name': [pkg_1, pkg_2, ...], ...}
        pkg_dict: OrderedDict = self.get_fit_packages(omix)
        for ml_key, pkg_list in pkg_dict.items():
          # `LogisticRegression` -> `LR`
          if abbreviate: ml_key = abbreviation_dict[ml_key]

          # Register ml_key if not exists
          if ml_key not in col_labels: col_labels.append(ml_key)

          mat_key = (sf_key, ml_key)
          # Register mat_key if not exists
          if mat_key not in matrix_dict: matrix_dict[mat_key] = []

          matrix_dict[mat_key].extend(pkg_list)

    return row_labels, col_labels, matrix_dict

  def report(self, metrics=('AUC', 'F1')):
    console.section('Pipeline Report')
    row_labels, col_labels, matrix_dict = self.get_pkg_matrix()
    for sf_key in row_labels:
      console.show_info(f'Feature selection method: {sf_key}')
      for ml_key in col_labels:
        console.supplement(f'Model: {ml_key}', level=3)
        pkg_list = matrix_dict[(sf_key, ml_key)]
        n_pkg = len(pkg_list)

        for key in metrics:
          values = [p[key] for p in pkg_list]
          mu = np.mean(values)
          CI1, CI2 = calc_CI(values, alpha=0.95, vmin=0., vmax=1.)
          info = f'Avg({key}) over {n_pkg} trials: {mu:.3f}'
          info += f', CI95% = [{CI1:.3f}, {CI2:.3f}]'
          console.supplement(info, level=4)

    print('-' * 79)

  def plot_matrix(self, fig_size=(5, 5)):
    metrics = ['AUC', 'Sensitivity', 'Selectivity',
               'Balanced Accuracy', 'Accuracy', 'F1']

    row_labels, col_labels, matrix_dict = self.get_pkg_matrix(abbreviate=True)

    # Generate matrices
    matrices = OrderedDict()
    value_dict = OrderedDict()
    for key in metrics:
      matrices[key] = np.zeros((len(row_labels), len(col_labels)))
      value_dict[key] = OrderedDict()
      for r, sf_key in enumerate(row_labels):
        value_dict[key][r] = OrderedDict()
        for c, ml_key in enumerate(col_labels):
          pkg_list = matrix_dict[(sf_key, ml_key)]
          values = [p[key] for p in pkg_list]
          matrices[key][r, c] = np.mean(values)
          value_dict[key][r][c] = values

    # Plot matrices
    from pictor.plotters.matrix_viewer import MatrixViewer

    row_labels = [rl.upper() for rl in row_labels]

    MatrixViewer.show_matrices(matrices, row_labels, col_labels, fig_size,
                               values=value_dict)

  def plot_calibration_curve(self, sf_method, *ml_methods, n_bins=10, **kwargs):
    from pictor.xomics.evaluation.calibration import Calibrator

    _, _, matrix_dict = self.get_pkg_matrix(abbreviate=True)

    true_dict, prob_dict = OrderedDict(), OrderedDict()
    for key in ml_methods:
      mat_key = (sf_method, key)
      assert mat_key in matrix_dict, f'No data for {mat_key}'
      pkg_list = matrix_dict[mat_key]
      true_dict[key] = [pkg.ROC.targets for pkg in pkg_list]
      prob_dict[key] = [pkg.probabilities[:, 1] for pkg in pkg_list]

    c = Calibrator(prob_dict, true_dict, **kwargs)
    c.plot_calibration_curve(n_bins=n_bins)

  # endregion: Public Methods

  # region: Pipeline Methods

  @staticmethod
  def grid_search(omix, sf_methods=('lasso', 'sig'),
                  ml_methods=('lr', 'svm', 'dt', 'rf', 'xgb'),
                  k=None, sf_repeats=10, ml_repeats=10, config_str='',
                  ignore_warnings=1, **kwargs):  # TODO --------------------
    """Grid search for feature selection methods and machine learning methods.
    """
    pi = Pipeline(omix, ignore_warnings=ignore_warnings, save_models=0)

    # (1) Load pre-defined settings
    if sf_methods == '':
      pass

    if ml_methods == '':
      pass

    # (1.1) Sanity check
    if isinstance(sf_methods, str): sf_methods = [sf_methods]
    if isinstance(ml_methods, str): ml_methods = [ml_methods]

    # (2) Create subspaces

    # (-1) Report
    pi.report()

    return omix

  # endregion: Pipeline Methods

  # region: Evaluation

  def get_best_pipeline(self, rank=1, verbose=1, reducer=None, model=None):
    ranking = self.pipeline_ranking

    if reducer not in (None, ''):
      ranking = [r for r in ranking if r[1].name.lower() == reducer]
      assert len(ranking) > 0

    if model not in (None, ''):
      ranking = [r for r in ranking if r[2].name.lower() == model]
      assert len(ranking) > 0

    selected_dr: DREngine = ranking[rank - 1][1]
    selected_pkg: FitPackage = ranking[rank - 1][2]

    if verbose:
      MAX_RANK = 10
      rank_str = ', '.join([f'[{i+1}{"*" if i + 1 == rank else ""}] {r[0]:.3f}'
                            for i, r in enumerate(ranking[:MAX_RANK])])
      console.show_info(f'AUC ranking: {rank_str}')
      self._report_dr_ml(selected_dr, selected_pkg)

    return selected_dr, selected_pkg

  def evaluate_best_pipeline(self, omix: Omix, rank=1, verbose=1) -> FitPackage:
    dr, pkg = self.get_best_pipeline(rank, verbose=verbose)

    omix_reduced = dr.reduce_dimension(omix)
    pkg = pkg.evaluate(omix_reduced)
    return pkg

  def _report_dr_ml(self, dr, pkg):
    console.show_info('Pipeline components:')
    console.supplement(f'Dimension reducer: {dr.__class__.__name__}', level=2)
    console.supplement(f'Machine learning model: {pkg.model_name}',
                       level=2)

  # endregion: Evaluation
