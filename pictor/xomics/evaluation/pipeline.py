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
    """[..., (AUC, DR_Model, ML_PKG), ...]"""
    ranking = []
    for _, omix_list in self.sub_space_dict.items():
      # Gather dimension reducers, in case any omix is duplicated thus has no dr
      # ... even save_model is True
      # reducers = [omix.dimension_reducer for omix in omix_list
      #            if isinstance(omix, Omix)
      #             and omix.dimension_reducer is not None]
      # shared_reducer = reducers[0] if len(reducers) == 1 else None

      for omix in omix_list:
        nested_dr = isinstance(omix, tuple) and isinstance(omix[0], Omix)
        if not nested_dr: assert isinstance(omix, Omix)

        pkg_dict: OrderedDict = self.get_fit_packages(omix)

        for _, pkg_list in pkg_dict.items():
          for pkg in pkg_list:
            if nested_dr:
              reducer = pkg.reducers
            else: reducer = omix.dimension_reducer

            # if reducer is None: reducer = shared_reducer
            assert reducer is not None

            ranking.append((pkg['AUC'], reducer, pkg))

    return sorted(ranking, key=lambda x: x[0], reverse=True)

  # endregion: Properties

  # region: Feature Selection

  def create_sub_space(self, method: str='*', repeats=1, nested=0,
                       show_progress=1, **kwargs):
    """Reduce dimension of self.omix and put the result into sub_space_dict.

    Args:
        method (str): Feature selection method. Defaults to '*'.
        repeats (int): Number of repeats. Defaults to 1.
        nested (bool): Whether to use nested cross-validation.
           When set to True, dimension reduction process will be delayed
           to the internal cross-validation stage. Defaults to False.
        show_progress (bool): Whether to report progress. Defaults to 1.
    """
    # (0) Initialization
    method = method.lower()
    prompt = '[FEATURE SELECTION] >>'

    # (0.1) Set default settings
    if 'save_model' not in kwargs: kwargs['save_model'] = self.save_models

    # (0.2) Initialize bag if not exists
    key = (method, tuple(sorted(tuple(kwargs.items()), key=lambda x: x[0])))
    if key not in self.sub_space_dict: self.sub_space_dict[key] = []

    # (0.3) Show progress bar if required
    if show_progress: console.show_status(
      f'Creating sub-feature space using `{method}` ...', prompt=prompt)

    # (1) Create sub-spaces for `repeat` times
    for i in range(repeats):
      # (1.1) Show progress bar if required
      if show_progress: console.print_progress(i, repeats)

      if nested:
        # When performing nested dimension reduction, save_model is
        #  automatically set to True
        kwargs.pop('save_model', None)
        # PROTOCOL for nested dimension reduction
        omix_sub = (self.omix.duplicate(), method, kwargs)
      else:
        shadow_omix = omix_sub if method == 'pca' and i > 0 else None
        omix_sub = self.omix.select_features(method, shadow_omix=shadow_omix,
                                             **kwargs)

      # (1.-1) Put sub-space-omix into sub_space_dict
      self.sub_space_dict[key].append(omix_sub)

    # (-1) Report progress if required
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
      f'Traverse through {N} subspaces '
      f'(repeat={repeats}{nested_suffix}) using {model} ...',
      prompt=prompt)

    for i, omix in enumerate(sub_spaces):
      if show_progress: console.print_progress(i, N)

      # (2.0) Get package dictionary
      pkg_dict = self.get_fit_packages(omix)

      model_name = str(model)
      if model_name not in pkg_dict: pkg_dict[model_name] = []

      # (2.1) tune hyper-parameters
      if not nested:
        if not isinstance(omix, Omix) and isinstance(omix, tuple):
          raise AssertionError(r'!! nested dimension reduction should be used '
                               r'with callable nested hp tuning')
        hp = model.tune_hyperparameters(omix, verbose=verbose)
      else: hp = None

      # (2.2) Repeatedly fit model on omix
      for _ in range(repeats):
        pkg = model.fit_k_fold(omix, hp=hp, save_models=self.save_models,
                               nested_ml=nested, **kwargs)
        pkg_dict[model_name].append(pkg)

    if show_progress: console.show_status(
      f'Traversed through {N} subspaces for {repeats} times.', prompt=prompt)

  # endregion: Fitting

  # region: Public Methods

  def get_fit_packages(self, omix: Omix) -> OrderedDict:
    """Format: {'model_name': [pkg_1, pkg_2, ...], ...}"""
    if isinstance(omix, tuple):
      assert len(omix) == 3
      omix = omix[0]
    return omix.get_from_pocket(
      'pp::fit_packages::24ma14', initializer=lambda: OrderedDict(), local=True)

  def get_pkg_matrix(self, abbreviate=False, omix_refit=None, omix_test=None,
                     random_state=None, verbose=1):
    """Get package matrix for each reducer-model combination.
       - If omix_refit is provided, each combination will be refitted.
       - If omix_test is provided, each combination will be validated on it.
         Otherwise, previous evaluation results will be used to fill the matrix.

       Note: If omix_refit or omix_test is provided, the result is not
             deterministic.
    """
    from pictor.xomics.ml import abbreviation_dict

    row_labels, col_labels, matrix_dict = [], [], {}

    # For each sub-space
    for key, omix_list in self.sub_space_dict.items():
      # (1) Generate reducer key
      # key = ('sf_method_name', (('arg_1', arg_1_value), ...))
      sf_key = key[0]

      # Add feature number to specific methods
      if sf_key in ('pca', 'mrmr', 'sig', 'pval'):
        arg, val = key[1][0]
        if arg in ('n_components', 'k'):
          sf_key += f'-{val}'

      # Register sf_key if not exists
      if sf_key not in row_labels: row_labels.append(sf_key)

      # (2) Traverse through models
      for omix in omix_list:
        # {'model_name': [pkg_1, pkg_2, ...], ...}
        pkg_dict: OrderedDict = self.get_fit_packages(omix)
        for ml_key, pkg_list in pkg_dict.items():
          # (2.1) Settle ml_key
          # `LogisticRegression` -> `LR`
          if abbreviate: ml_key = abbreviation_dict[ml_key]

          # Register ml_key if not exists
          if ml_key not in col_labels: col_labels.append(ml_key)

          # (2.2) Initialize mat slot if not exists
          mat_key = (sf_key, ml_key)
          # Register mat_key if not exists
          if mat_key not in matrix_dict: matrix_dict[mat_key] = []

          # (2.3) Fill in the matrix
          # (2.3.1) Test pipeline on omix_test if provided
          if isinstance(omix_test, Omix):
            # TODO: duplicated with pipeline_ranking logic
            pkg_list = [self.evaluate_pipeline(
              omix_test, omix_refit=omix_refit, pkg=pkg,
              reducer=(pkg.reducers if isinstance(omix, tuple)
                       else omix.dimension_reducer),
              random_seed=random_state, verbose=verbose)
              for pkg in pkg_list]

          # (2.3.-1) Put package list into slot
          matrix_dict[mat_key].extend(pkg_list)

    return row_labels, col_labels, matrix_dict

  def refit_pipeline(self, omix: Omix, random_seed=None):
    pass

  def report(self, metrics=None):
    if metrics is None:
      if self.omix.targets_are_numerical: metrics = ('MAE',)
      else: metrics = ('AUC', 'F1')

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

          CI1, CI2 = calc_CI(values, alpha=0.95, key=key)
          info = f'Avg({key}) over {n_pkg} trials: {mu:.3f}'
          info += f', CI95% = [{CI1:.3f}, {CI2:.3f}]'
          console.supplement(info, level=4)

    print('-' * 79)

  def plot_matrix(self, fig_size=(5, 5), omix_refit=None, omix_test=None,
                  random_state=None, verbose=0):

    if self.omix.targets_are_numerical: metrics = ['MAE']
    else: metrics = ['AUC', 'Sensitivity', 'Selectivity',
                     'Balanced Accuracy', 'Accuracy', 'F1']

    row_labels, col_labels, matrix_dict = self.get_pkg_matrix(
      abbreviate=True, omix_refit=omix_refit, omix_test=omix_test,
      random_state=random_state, verbose=verbose)

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

    cmap = 'Blues' if omix_test is None else 'Oranges'
    MatrixViewer.show_matrices(matrices, row_labels, col_labels, fig_size,
                               values=value_dict, cmap=cmap)

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

  # endregion: Pipeline Methods

  # region: Evaluation

  def get_best_pipeline(self, rank=1, verbose=1, reducer=None, model=None):
    ranking = self.pipeline_ranking
    nested_dr = isinstance(ranking[0][1], (tuple, list))

    if reducer not in (None, ''):
      if nested_dr:
        ranking = [r for r in ranking if r[1][0].name.lower() == reducer]
      else:
        ranking = [r for r in ranking if r[1].name.lower() == reducer]
      assert len(ranking) > 0

    if model not in (None, ''):
      ranking = [r for r in ranking if r[2].name.lower() == model]
      assert len(ranking) > 0

    # Actually, when nested dimension reduction is used, selected_dr will be a
    # list of dimension reducers
    selected_dr: DREngine = ranking[rank - 1][1]
    selected_pkg: FitPackage = ranking[rank - 1][2]

    if verbose:
      MAX_RANK = 10
      rank_str = ', '.join([f'[{i+1}{"*" if i + 1 == rank else ""}] {r[0]:.3f}'
                            for i, r in enumerate(ranking[:MAX_RANK])])
      console.show_info(f'AUC ranking: {rank_str}')
      self._report_dr_ml(selected_dr, selected_pkg)

    return selected_dr, selected_pkg

  def evaluate_best_pipeline(self, omix: Omix, omix_refit: Omix = None, rank=1,
                             verbose=1, model=None, reducer=None) -> FitPackage:
    dr, pkg = self.get_best_pipeline(rank, verbose=verbose, reducer=reducer,
                                     model=model)

    return self.evaluate_pipeline(omix, omix_refit=omix_refit,
                                  pkg=pkg, reducer=dr, verbose=verbose)


  def evaluate_pipeline(self, omix: Omix, omix_refit: Omix = None,
                        pkg: FitPackage = None, reducer: DREngine = None,
                        random_seed=None, verbose=1) -> FitPackage:
    # (0) Sanity check
    if omix_refit is None:
      assert isinstance(pkg, FitPackage), '!! Either omix_refit or pkg is required'

    # (A) Non-refit
    if omix_refit is None:
      # (A-1) Reduce dimension if valid reducer is provided
      #  Otherwise dimension reduction will be performed
      #  before each model is called
      if isinstance(reducer, (tuple, list)): omix_reduced = omix
      else: omix_reduced = reducer.reduce_dimension(omix)

      new_pkg = pkg.evaluate(omix_reduced)
      return new_pkg

    # (B) Refit TODO: see ml_engine.fit_k_fold
    from pictor.xomics.ml import SK_TO_OMIX_DICT

    if isinstance(reducer, (tuple, list)): reducer = reducer[0]
    ModelClass = SK_TO_OMIX_DICT[pkg.models[0].__class__]

    if verbose:
      model_name = ModelClass.__name__
      console.show_status(f'Refitting {reducer}->{model_name} pipeline ...')

    # (B-1) Fit reducer and reduce dimension
    reducer.fit_reducer(omix_refit, random_state=random_seed, exclusive=False,
                        verbose=verbose)
    omix_refit_reduced = reducer.reduce_dimension(omix_refit)

    # (B-2) Fit machine learning model
    model = ModelClass()
    hp = model.tune_hyperparameters(
      omix_refit_reduced, verbose=verbose, random_state=random_seed)
    sk_model = model.fit(omix_refit_reduced, hp=hp, random_state=random_seed)

    # (B-3) Test model on test data
    omix_reduced = reducer.reduce_dimension(omix)
    prob = sk_model.predict_proba(omix_reduced.features)
    pred = sk_model.predict(omix_reduced.features)

    return FitPackage.pack(pred, prob, omix, hp=hp)

  def _report_dr_ml(self, dr, pkg):
    console.show_info('Pipeline components:')

    if isinstance(dr, list): dr = dr[0]
    console.supplement(f'Dimension reducer: {dr.__class__.__name__}', level=2)
    console.supplement(f'Machine learning model: {pkg.model_name}',
                       level=2)

  # endregion: Evaluation
