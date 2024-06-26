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
"""TODO: This module is an intermediate version during the refactoring process.
"""
from .dr_engine import DREngine
from pictor.xomics.ml.lasso import Lasso
from pictor.xomics.omix import Omix
from roma import console

import numpy as np



class LASSO(Lasso, DREngine):
  """Least Absolute Shrinkage and Selection Operator (LASSO) for dimension
  reduction."""

  TYPE = DREngine.Types.Selector

  def __init__(self, verbose: int = 0, ignore_warnings=False, n_jobs=5,
               standardize: bool = True, **configs):
    # Call parent constructors
    Lasso.__init__(self, verbose=verbose, ignore_warnings=ignore_warnings,
                   n_jobs=n_jobs)
    DREngine.__init__(self, standardize=standardize, **configs)


  def _fit_reducer(self, omix: Omix, **kwargs):
    # (0) get settings
    verbose = kwargs.get('verbose', self.verbose)
    threshold = kwargs.get('threshold', 0.001)
    alpha = kwargs.get('alpha', None)

    # (1) Tune hyperparameters
    if alpha is None:
      alpha = self.tune_alpha(omix, **kwargs)
      hp = {'alpha': alpha}
    else:
      hp = {'alpha': alpha}

    # (2) Fit model and get importance
    lasso = self.fit(omix, hp=hp, **kwargs)
    importance = np.abs(lasso.coef_)

    if verbose > 1: self.plot_importance(importance, omix.feature_labels)

    # (3) Return reducer, and indices
    indices = np.where(importance > threshold)[0]
    if verbose > 0: console.show_status(f'Selected {len(indices)} features.')
    return lasso, indices
