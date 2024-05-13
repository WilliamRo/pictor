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
from roma import console
from roma import Nomear

import numpy as np
import os
import time
import warnings



class MLEngine(Nomear):

  SK_CLASS = None
  DEFAULT_HP_SPACE = None
  DEFAULT_HP_MODEL_INIT_KWARGS = {}
  EXTRA_FIT_KWARGS = {}

  def __init__(self, verbose: int = 0, ignore_warnings=False, n_jobs=5):
    # 0. Ignore warnings if required
    if ignore_warnings:
      warnings.simplefilter('ignore')
      os.environ["PYTHONWARNINGS"] = "ignore"
      console.show_status('Warning Ignored.', prompt='[MLEngine] >>')

    self.verbose = verbose
    self.n_jobs = n_jobs

  # region: Properties

  @property
  def best_hp(self): return self.get_from_pocket(
      'best_hp', key_should_exist=True, local=True)

  @best_hp.setter
  def best_hp(self, val): self.put_into_pocket(
    'best_hp', val, exclusive=False, local=True)

  # endregion: Properties

  # region: Hyperparameter Tuning

  def tune_hyperparameters(self, omix: Omix, **kwargs) -> dict:
    prompt = '[TUNE] >>'
    from sklearn.model_selection import KFold

    # (0) get settings
    random_state = kwargs.get('random_state', None)

    n_splits = kwargs.get('n_splits', 5)
    n_jobs = kwargs.get('n_jobs', self.n_jobs)
    strategy = kwargs.get('strategy', 'grid')
    hp_space = kwargs.get('hp_space', self.DEFAULT_HP_SPACE)
    hp_model_init_kwargs = kwargs.get('hp_model_init_kwargs',
                                      self.DEFAULT_HP_MODEL_INIT_KWARGS)

    verbose = kwargs.get('verbose', self.verbose)

    # (0.5) Construct hp_space
    if not isinstance(hp_space, list): hp_space = [hp_space]
    # Sanity check
    for hp_dict in hp_space:
      assert isinstance(hp_dict, dict), '!! hp_space should be a list of dict.'

    # (1) Initiate a model
    model = self.SK_CLASS(random_state=random_state, **hp_model_init_kwargs)

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

        search_cv = GridSearchCV(model, hp_dict, verbose=verbose, cv=kf,
                                 n_jobs=n_jobs)
      elif strategy in ('rand', 'random', 'random_search'):
        from sklearn.model_selection import RandomizedSearchCV

        n_iter = kwargs.get('n_iter', 10)  # <= Number of iterations
        search_cv = RandomizedSearchCV(
          model, hp_dict, cv=kf, n_iter=n_iter, verbose=verbose,
          n_jobs=n_jobs, random_state=random_state)
      else: raise ValueError(f'!! Unknown strategy: {strategy}')

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
    hp = kwargs.get('hp', self.best_hp)

    hp.update(self.EXTRA_FIT_KWARGS)

    # (1) Initiate model
    model = self.SK_CLASS(random_state=random_state, **hp)

    # (2) Fit model
    model.fit(omix.features, omix.targets)

    return model

  def fit_k_fold(self, omix: Omix, **kwargs):
    prompt = '[K_FOLD_FIT] >>'
    # (0) Get settings
    random_state = kwargs.get('random_state', None)

    hp = kwargs.get('hp', None)
    n_splits = kwargs.get('n_splits', 5)

    verbose = kwargs.get('verbose', self.verbose)

    # (1) Tune hyperparameters if required
    if hp is None: hp = self.tune_hyperparameters(omix, verbose=verbose,
                                                  random_state=random_state)

    # (2) Fit data in k-fold manner
    if verbose > 0:
      console.show_status(f'{n_splits}-fold fitting with seed = {random_state}',
                          prompt=prompt)

    models, prob_list, pred_list = [], [], []
    k_fold_data, om_whole = omix.get_k_folds(
      k=n_splits, shuffle=True, random_state=random_state, return_whole=True)

    for i, (om_train, om_test) in enumerate(k_fold_data):
      if verbose > 3: om_test.report()

      lr = self.fit(om_train, hp=hp, random_state=random_state)

      prob = lr.predict_proba(om_test.features)
      pred = lr.predict(om_test.features)

      prob_list.append(prob)
      pred_list.append(pred)
      models.append(pred)

    # probabilities.shape = (n_samples, n_classes)
    probabilities = np.concatenate(prob_list, axis=0)
    predictions = np.concatenate(pred_list)

    if verbose > 0:
      console.show_status('Fitting completed.', prompt=prompt)

    # (3) Analyze results if required
    if kwargs.get('cm', False):
      from pictor.xomics.evaluation.confusion_matrix import ConfusionMatrix

      cm = ConfusionMatrix(num_classes=2, class_names=omix.target_labels)
      cm.fill(predictions, om_whole.targets)

      if kwargs.get('print_cm', False):
        console.show_info(f'Confusion Matrix ({omix.data_name}):')
        self.print(cm.make_matrix_table())

      console.show_info(f'Evaluation Result ({omix.data_name}):')
      self.print(cm.make_result_table(decimal=4, class_details=True))

      if kwargs.get('plot_cm', False): cm.sklearn_plot()

    if kwargs.get('auc', False):
      from pictor.xomics.evaluation.roc import ROC

      # Calculate AUC
      auc = ROC.calc_auc(probabilities[:, 1], om_whole.targets)
      console.show_info(f'AUC ({omix.data_name}) = {auc:.3f}')

      if kwargs.get('plot_roc', False):
        ROC.plot_roc(probabilities[:, 1], om_whole.targets)

    # (-1) Return the fitted models and probabilities
    return models, probabilities, predictions

  # endregion: Machine Learning

  # region: Results Analysis



  # endregion: Results Analysis

  # region: MISC

  def print(self, content: str):
    # roma.console.write_line does not fit cm, this is a workaround
    from sys import stdout
    stdout.write("\r{}\n".format(content))
    stdout.flush()

  # endregion: MISC

