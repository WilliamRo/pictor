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
# ==-=======================================================================-===
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.confusion_matrix import ConfusionMatrix
from pictor.xomics.evaluation.roc import ROC

from roma import console
from roma import Nomear

import numpy as np
import os
import time
import warnings



class MLEngine(Nomear):

  SK_CLASS = None
  IS_CLASSIFIER = True
  DEFAULT_HP_SPACE = None
  DEFAULT_HP_MODEL_INIT_KWARGS = {}
  EXTRA_FIT_KWARGS = {}

  DISABLE_PARALLEL = False

  abbreviation = None

  def __init__(self, verbose: int = 0, ignore_warnings=False, n_jobs=5):
    # 0. Ignore warnings if required
    if ignore_warnings:
      warnings.simplefilter('ignore')
      os.environ["PYTHONWARNINGS"] = "ignore"
      console.show_status('Warning Ignored.', prompt='[MLEngine] >>')

    self.verbose = verbose
    self.n_jobs = 1 if self.DISABLE_PARALLEL else n_jobs

  # region: Properties

  @property
  def best_hp(self): return self.get_from_pocket(
      'best_hp', key_should_exist=True, local=True)

  @best_hp.setter
  def best_hp(self, val): self.put_into_pocket(
    'best_hp', val, exclusive=False, local=True)

  @property
  def name(self): return self.__class__.__name__

  # endregion: Properties

  # region: Hyperparameter Tuning

  def tune_hyperparameters(self, omix: Omix, **kwargs) -> dict:
    """Tune hyperparameters using grid search or random search on omix
    in a k-fold manner."""
    prompt = '[TUNE] >>'
    from sklearn.model_selection import KFold

    # (0) get settings
    random_state = kwargs.get('random_state', None)

    n_splits = kwargs.get('n_splits', 5)
    n_jobs = kwargs.get('n_jobs', self.n_jobs)
    strategy = kwargs.get('strategy', 'grid')

    hp_space = kwargs.get('hp_space', None)
    if hp_space is None: hp_space = self.DEFAULT_HP_SPACE

    hp_model_init_kwargs = kwargs.get('hp_model_init_kwargs',
                                      self.DEFAULT_HP_MODEL_INIT_KWARGS)

    verbose = kwargs.get('verbose', self.verbose)
    grid_repeats = kwargs.get('grid_repeats', 1)

    if len(hp_space) == 0: return {}

    # (0.5) Construct hp_space
    if not isinstance(hp_space, list): hp_space = [hp_space]
    # Sanity check
    for hp_dict in hp_space:
      assert isinstance(hp_dict, dict), '!! hp_space should be a list of dict.'

    # (1) Initiate a model
    if not omix.targets_are_numerical:
      hp_model_init_kwargs['random_state'] = random_state
    model = self.SK_CLASS(**hp_model_init_kwargs)

    # (2) Search for the best hyperparameters based on cross-validation
    if verbose > 0:
      if random_state is not None:
        console.show_status(f'Random seed set to {random_state}', prompt=prompt)
      console.show_status(f'Tuning hyperparameters using {strategy} search...',
                          prompt=prompt)
      console.show_status(
        f'ConvergenceWarnings may appear if warning is not ignored.',
        prompt=prompt)

    time_start = time.time()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Find the best hyperparameters for each setting
    searchers = []
    for hp_dict in hp_space:
      if strategy in ('grid', 'grid_search'):
        from sklearn.model_selection import GridSearchCV

        search_cvs = [GridSearchCV(model, hp_dict, verbose=verbose, cv=kf,
                                   n_jobs=n_jobs) for _ in range(grid_repeats)]
      elif strategy in ('rand', 'random', 'random_search'):
        from sklearn.model_selection import RandomizedSearchCV

        n_iter = kwargs.get('n_iter', 10)  # <= Number of iterations
        search_cvs = [RandomizedSearchCV(
          model, hp_dict, cv=kf, n_iter=n_iter, verbose=verbose,
          n_jobs=n_jobs, random_state=random_state)]
      else: raise ValueError(f'!! Unknown strategy: {strategy}')

      for search_cv in search_cvs:
        search_cv.fit(omix.features, omix.targets)
        searchers.append(search_cv)

    # (3) Return the best hyperparameters
    searchers = sorted(searchers, key=lambda x: x.best_score_, reverse=True)
    best_searcher = searchers[0]
    self.best_hp = best_searcher.best_params_
    if verbose > 0:
      console.show_status(f'Best hyperparameters: {self.best_hp}',
                          prompt=prompt)
      console.show_status(f'Best score: {best_searcher.best_score_:.4f}',
                          prompt=prompt)

      elapsed_time = time.time() - time_start
      console.show_status(f'Elapsed time: {elapsed_time:.2f} seconds',
                          prompt=prompt)
    return self.best_hp

  # endregion: Hyperparameter Tuning

  # region: Machine Learning

  def fit(self, omix: Omix, **kwargs):
    # (0) get settings
    random_state = kwargs.get('random_state', None)
    hp = kwargs.get('hp', None)
    if hp is None: hp = self.best_hp

    hp.update(self.EXTRA_FIT_KWARGS)

    # (1) Initiate model
    if not omix.targets_are_numerical: hp['random_state'] = random_state
    model = self.SK_CLASS(**hp)

    # (2) Fit model
    model.fit(omix.features, omix.targets)

    return model

  def fit_k_fold(self, omix: Omix, nested_ml=False, **kwargs) -> 'FitPackage':
    """Fit the model in a k-fold manner.

    If omix is not an instance of Omix, it should be a tuple in a form of
    (omix: Omix, method: str, kwargs: dict). In such cases, `nested` should
    be True and feature reduction will be performed inside each inner CV.
    """
    prompt = '[K_FOLD_FIT] >>'
    # (0) Get settings
    random_state = kwargs.get('random_state', None)

    hp = kwargs.get('hp', None)
    hp_space = kwargs.get('hp_space', None)
    n_splits = kwargs.get('n_splits', 5)
    shuffle = kwargs.get('shuffle', True)

    verbose = kwargs.get('verbose', self.verbose)

    nested_prefix = 'Nested ' if nested_ml else ''
    if verbose > 0:
      console.section(f'{nested_prefix}K-fold fitting using {self}')

    # (1) Tune hyperparameters if necessary
    if hp is None and not nested_ml:
      if not isinstance(omix, Omix) and isinstance(omix, tuple):
        raise AssertionError(r'!! nested dimension reduction should be used '
                             r'with callable nested hp tuning')
      hp = self.tune_hyperparameters(
        omix, verbose=verbose, random_state=random_state, hp_space=hp_space)

    # (2) Fit data in k-fold manner
    if verbose > 0: console.show_status(
      f'{nested_prefix}{n_splits}-fold fitting with seed = {random_state}',
      prompt=prompt)

    # (2.1) Preparation
    models, prob_list, pred_list, fold_pkgs, reducers = [], [], [], [], []

    # (2.1.1) Unpack omix if using nested dimension reduction
    dr_key, dr_kwargs, nested_dr = None, None, False
    if not isinstance(omix, Omix):
      assert isinstance(omix, tuple) and len(omix) == 3
      omix, dr_key, dr_kwargs = omix
      nested_dr = True

    # (2.1.2) Get k-fold data
    k_fold_data, om_whole = omix.get_k_folds(
      k=n_splits, shuffle=shuffle, random_state=random_state, return_whole=True)

    # (2.2) Run outer loop of CV
    for i, (om_train, om_test) in enumerate(k_fold_data):
      # (2.2.1) Report test data details if required
      if verbose > 3: om_test.report()

      # (2.2.2) Perform dimension reduction if specified
      if nested_dr:
        # (2.2.2.0) Report progress if required
        if verbose > 0: console.show_status(
          f'Reducing dimension for fold-{i+1}/{n_splits}...',)

        # (2.2.2.1) Perform dimension reduction on om_train
        om_train: Omix = om_train.select_features(
          dr_key, verbose=verbose, save_model=True, **dr_kwargs)

        # (2.2.2.2) Perform dimension reduction on om_test
        reducer = om_train.dimension_reducer
        om_test = reducer.reduce_dimension(om_test)

        reducers.append(reducer)

      # (2.2.3) Tune parameters on om_train if necessary
      if nested_ml:
        if verbose > 0: console.show_status(
          f'Tuning hyperparameters for fold-{i+1}/{n_splits}...',)
        hp = self.tune_hyperparameters(
          om_train, verbose=verbose, random_state=random_state,
          hp_space=hp_space)

      # (2.2.4) Fit the model
      model = self.fit(om_train, hp=hp, random_state=random_state)

      # (2.2.5) Evaluate model on test data
      pred = model.predict(om_test.features)
      if not omix.targets_are_numerical:
        prob = model.predict_proba(om_test.features)
      else:
        prob = pred

      # (2.2.-1) Pack the results
      prob_list.append(prob)
      pred_list.append(pred)
      models.append(model)

      fold_pkgs.append(FitPackage.pack(pred, prob, om_test, hp=hp))

    # probabilities.shape = (n_samples, n_classes)
    probabilities = np.concatenate(prob_list, axis=0)
    predictions = np.concatenate(pred_list)

    if verbose > 0:
      console.show_status('Fitting completed.', prompt=prompt)

    # (3) Analyze results if required
    if not kwargs.get('save_models', False): models = ()
    package = FitPackage.pack(predictions, probabilities, om_whole, models, hp,
                              sub_packages=fold_pkgs, reducers=reducers)

    # This is for the convenience of evaluation
    package.put_into_pocket('sample_labels', om_whole.sample_labels,
                            local=True)

    package.report(print_cm=kwargs.get('print_cm', False),
                   print_cm_table=kwargs.get('cm', False),
                   plot_cm=kwargs.get('plot_cm', False),
                   print_misclassified=kwargs.get('mi', False),
                   print_auc=kwargs.get('auc', False),
                   plot_roc=kwargs.get('plot_roc', False),
                   show_signature=kwargs.get('show_signature', False),
                   omix=om_whole, mae=kwargs.get('mae', False),
                   ra=kwargs.get('ra', False),
                   mi_remap=lambda i: om_whole.sample_labels[i])

    # (-1) Return the fitted models and probabilities
    return package

  # endregion: Machine Learning

  # region: MISC

  def plot_learning_curve(self, omix: Omix, **kwargs):
    from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

    import matplotlib.pyplot as plt
    import numpy as np

    # Get settings
    random_state = kwargs.get('random_state', None)
    n_splits = kwargs.get('n_splits', 5)

    hp = kwargs.get('hp', None)
    if hp is None:
      if not self.in_pocket('best_hp'):
        self.tune_hyperparameters(omix, **kwargs)
      hp = self.best_hp

    cv = ShuffleSplit(n_splits=n_splits,
                      test_size=0.2,
                      random_state=random_state)

    common_params = {
      "X": omix.features,
      "y": omix.targets,
      "train_sizes": np.linspace(0.1, 1.0, 10),
      "cv": cv,
      "score_type": "both",
      # "n_jobs": 4,
      "line_kw": {"marker": "o"},
      "std_display_style": "fill_between",
      "score_name": "Accuracy",
    }

    model = self.SK_CLASS(**hp)
    LearningCurveDisplay.from_estimator(model, **common_params)
    ax = plt.gca()
    handles, label = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Training Score", "Test Score"])
    ax.set_title(f"Learning Curve for {self}")
    plt.show()

  # endregion: MISC

  # region: Overriding

  def __str__(self):
    if self.abbreviation is not None: return self.abbreviation
    return str(self.__class__.__name__)

  # endregion: Overriding



class FitPackage(Nomear):

  def __init__(self, hyper_parameters: dict,
               confusion_matrix: ConfusionMatrix,
               models: list, roc: ROC,
               probabilities: np.ndarray,
               reducers=(),
               sub_packages=None,
               targets=None, mae=None,
               model_is_regressor=False):
    self.hyper_parameters = hyper_parameters
    self.confusion_matrix: ConfusionMatrix = confusion_matrix
    self.models = models
    self.ROC = roc
    self.probabilities = probabilities
    self.sub_packages = sub_packages
    self.model_is_regressor = model_is_regressor

    self.reducers = reducers

    # Regressor metrics
    self.targets = targets
    self.mae = mae

  # region: Properties

  @property
  def model_name(self): return self.models[0].__class__.__name__

  @property
  def pearson_r(self):
    from scipy import stats
    _r, _pval = stats.pearsonr(self.targets, self.probabilities)
    return _r

  @property
  def ordered_targets_and_probs(self):
    sample_labels = self.get_from_pocket('sample_labels', key_should_exist=True)
    sample_labels = list(sample_labels)

    # Sort targets and probabilities by sample_labels
    sorted_sample_labels = sorted(sample_labels)
    sorted_indices = [sample_labels.index(i) for i in sorted_sample_labels]

    return self.targets[sorted_indices], self.probabilities[sorted_indices]

  # endregion: Properties

  # region: Overriding

  def __getitem__(self, item):
    if isinstance(item, str):
      item_l = item.lower()

      # Sample labels is set here for avoiding errors
      if item == 'sample_labels':
        if self.in_pocket(item): return self.get_from_pocket(item)
        else: raise ValueError('!! sample_labels is not in pocket.')

      # For regressor results
      if item_l == 'r': return self.pearson_r

      # Metrics like MAE is recorded in constructor
      # TODO: workaround for saved pkgs of old versions which has no
      #   'model_is_regressor' attribute
      if hasattr(self, 'model_is_regressor') and self.model_is_regressor:
        return self.__getattribute__(item_l)

      if item_l in ('auc', 'roc_auc'): return self.ROC.auc
      else: return self.confusion_matrix[item]
    else: raise ValueError('!! item should be a string.')

  # endregion: Overriding

  # region: Public Methods

  def predict_proba(self, X):
    if len(self.reducers) == len(self.models):
      return np.mean(
        [model.predict_proba(reducer.reduce_dimension(X))
         for reducer, model in zip(self.reducers, self.models)], axis=0)

    return np.mean([model.predict_proba(X) for model in self.models], axis=0)

  def predict_values(self, X):
    if len(self.reducers) == len(self.models):
      return np.mean(
        [model.predict(reducer.reduce_dimension(X))
         for reducer, model in zip(self.reducers, self.models)], axis=0)

    return np.mean([model.predict(X) for model in self.models], axis=0)

  def predict(self, X, threshold=0.5):
    # For regressors
    if self.model_is_regressor:
      return self.predict_values(X)

    # For classifiers
    proba = self.predict_proba(X)
    return (proba[:, 1] > threshold).astype(int)

  def evaluate(self, omix, threshold=0.5) -> 'FitPackage':
    """Evaluate the performance of the model."""
    predictions = self.predict(omix.features, threshold=threshold)
    probabilities = self.predict_proba(omix.features)
    return FitPackage.pack(predictions, probabilities, omix)

  def report(self, print_cm=True, print_cm_table=True, plot_cm=False,
             print_misclassified=False, print_auc=True, plot_roc=False,
             mae=True, ra=False,
             show_signature=False, mi_remap=None, omix=None, prompt='>> '):
    """Report the evaluation results of the fit package.

    :Args
    - mi_remap: callable, remap the misclassified indices
    """
    if self.model_is_regressor:
      if mae: console.show_status(f'MAE = {self.mae:.3f}')

      if ra:
        from pictor.xomics.evaluation.reg_ana import RegressionAnalysis
        assert self.model_is_regressor, '!! This is not a regression model.'
        ra = RegressionAnalysis(self.targets, self.probabilities)
        ra.plot_scatter()

      return

    # (1) Confusion matrix related
    cm = self.confusion_matrix
    if print_cm: self.print(cm.make_matrix_table())

    if print_cm_table:
      self.print(cm.make_result_table(decimal=4, class_details=True))

    if print_misclassified:
      missed_indices = cm.missed_indices
      if callable(mi_remap):
        missed_indices = [mi_remap(i) for i in missed_indices]
      console.show_status(f'Miss-classified indices (start from 0, {len(missed_indices)} samples):',
                          prompt=prompt)
      missed_indices = sorted(missed_indices)
      console.supplement(missed_indices, level=2)

    if plot_cm: cm.sklearn_plot()

    # (2) ROC related
    if print_auc:
      console.show_info(f'AUC = {self.ROC.auc:.3f}')

    if plot_roc: self.ROC.plot_roc()

    # (3) Signature related
    if show_signature:
      assert isinstance(omix, Omix), '!! omix should be provided for signature.'
      sig: Omix = omix.duplicate(
        features=self.probabilities[:, 1].reshape(-1, 1),
        feature_labels=['Signature'])
      sig.show_in_explorer()

  @classmethod
  def pack(cls, predictions, probabilities, omix: Omix,
           models=(), hp={}, sub_packages=(), reducers=()) -> 'FitPackage':
    """Construct a FitPackage from an Omix object."""
    cm, roc = None, None
    mae = None

    if omix.targets_are_numerical:
      # (1) For regressors
      model_is_regressor = True

      mae = np.mean(np.abs(probabilities - omix.targets))
    else:
      # (2) For classifiers
      cm = ConfusionMatrix(num_classes=2, class_names=omix.target_labels)
      cm.fill(predictions, omix.targets)

      roc = ROC(probabilities[:, 1], omix.targets)
      model_is_regressor = False

    package = FitPackage(hyper_parameters=hp, models=models, roc=roc,
                         confusion_matrix=cm, probabilities=probabilities,
                         sub_packages=sub_packages, reducers=reducers,
                         model_is_regressor=model_is_regressor,
                         targets=omix.targets, mae=mae)
    return package

  # endregion: Public Methods

  # region: MISC

  def print(self, content: str):
    # roma.console.write_line does not fit cm, this is a workaround
    from sys import stdout
    stdout.write("\r{}\n".format(content))
    stdout.flush()

  # endregion: MISC

