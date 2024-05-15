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
from pictor.xomics.omix import Omix
from roma import console
from roma import Nomear

import os
import warnings
import numpy as np



class Pipeline(Nomear):
  """Omix-based pipeline for feature selection, fitting, and evaluation.
  """
  prompt = '[PIPELINE] >>'

  def __init__(self, omix: Omix, ignore_warnings=False):
    # 0. Ignore warnings if required
    if ignore_warnings:
      warnings.simplefilter('ignore')
      os.environ["PYTHONWARNINGS"] = "ignore"
      console.show_status('Warning Ignored.', prompt=self.prompt)

    self.omix: Omix = omix

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

  # endregion: Properties

  # region: Feature Selection

  def create_sub_space(self, method: str, repeats=1,
                       show_progress=0, **kwargs):
    method = method.lower()
    prompt = '[FEATURE SELECTION] >>'

    if method == 'pca': assert repeats == 1, "Repeat PCA makes no sense."

    # Initialize bag if not exists
    key = (method, tuple(sorted(tuple(kwargs.items()), key=lambda x: x[0])))
    if key not in self.sub_space_dict: self.sub_space_dict[key] = []

    if show_progress: console.show_status(
      f'Creating sub-feature space using `{method}` ...', prompt=prompt)

    for i in range(repeats):
      if show_progress: console.print_progress(i, repeats)
      omix_sub = self.omix.select_features(method, **kwargs)

      self.sub_space_dict[key].append(omix_sub)

    if show_progress: console.show_status(
      f'{repeats} sub-feature spaces created.', prompt=prompt)

  # endregion: Feature Selection

  # region: Fitting

  def fit_traverse_spaces(self, model: str, repeats=1,
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
    if show_progress: console.show_status(
      f'Traverse through {N} subspaces (repeat={repeats}) using {model} ...',
      prompt=prompt)

    for i, omix in enumerate(sub_spaces):
      if show_progress: console.print_progress(i, N)

      pkg_dict = self.get_fit_packages(omix)
      model_name = str(model)
      if model_name not in pkg_dict: pkg_dict[model_name] = []

      # (2.1) tune hyper-parameters
      hp = model.tune_hyperparameters(omix, verbose=verbose)

      # (2.2) Repeatedly fit model on omix
      for _ in range(repeats):
        pkg = model.fit_k_fold(omix, hp=hp, **kwargs)
        pkg_dict[model_name].append(pkg)

    if show_progress: console.show_status(
      f'Traversed through {N} subspaces for {repeats} times.', prompt=prompt)

  # endregion: Fitting

  # region: Public Methods

  def get_fit_packages(self, omix: Omix) -> OrderedDict:
    """Format: {'model_name': [pkg_1, pkg_2, ...], ...}"""
    return omix.get_from_pocket(
      'pp::fit_packages::24ma14', initializer=lambda: OrderedDict(), local=True)

  def get_pkg_matrix(self):
    row_labels, col_labels, matrix_dict = [], [], {}

    # For each sub-space
    for key, omix_list in self.sub_space_dict.items():
      sf_key = key[0]
      # Register sf_key if not exists
      if sf_key not in row_labels: row_labels.append(sf_key)

      for omix in omix_list:
        pkg_dict: OrderedDict = self.get_fit_packages(omix)
        for ml_key, pkg_list in pkg_dict.items():
          # Register ml_key if not exists
          if ml_key not in col_labels: col_labels.append(ml_key)

          mat_key = (sf_key, ml_key)
          # Register mat_key if not exists
          if mat_key not in matrix_dict: matrix_dict[mat_key] = []

          matrix_dict[mat_key].extend(pkg_list)

    return row_labels, col_labels, matrix_dict

  def report(self, metrics=('AUC', 'F1')):
    import scipy.stats as st

    row_labels, col_labels, matrix_dict = self.get_pkg_matrix()
    for sf_key in row_labels:
      console.show_info(f'Feature selection method: {sf_key}')
      for ml_key in col_labels:
        console.supplement(f'Model: {ml_key}', level=2)
        pkg_list = matrix_dict[(sf_key, ml_key)]
        n_pkg = len(pkg_list)

        for key in metrics:
          values = [p[key] for p in pkg_list]
          mu = np.mean(values)
          CI1, CI2 = st.t.interval(0.95, n_pkg-1, loc=mu, scale=st.sem(values))
          info = f'Avg({key}) over {n_pkg} trials: {mu:.3f}'
          info += f', CI95% = [{CI1:.3f}, {CI2:.3f}]'
          console.supplement(info, level=3)

    print('-' * 79)

  # endregion: Public Methods
