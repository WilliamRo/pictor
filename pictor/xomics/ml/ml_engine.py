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
from roma import Nomear



class MLEngine(Nomear):

  def __init__(self, omix: Omix, random_state=None):
    self.omix = omix
    self.random_state = random_state


  def tune_hyperparameters(self, model, hp_space: dict, n_splits=5,
                           strategy='grid', verbose=0, **kwargs) -> dict:
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import KFold

    # Sanity check
    assert isinstance(model, BaseEstimator), '!! model must be a sklearn estimator'

    # Search for the best hyperparameters based on cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    if strategy in ('grid', 'grid_search'):
      from sklearn.model_selection import GridSearchCV

      search_cv = GridSearchCV(model, hp_space, verbose=verbose, cv=kf)
    elif strategy in ('rand', 'random', 'random_search'):
      from sklearn.model_selection import RandomizedSearchCV

      n_iter = kwargs.get('n_iter', 10)
      search_cv = RandomizedSearchCV(
        model, hp_space, cv=kf, n_iter=n_iter, verbose=verbose)
    else: raise ValueError(f'!! Unknown strategy: {strategy}')

    search_cv.fit(self.omix.features, self.omix.targets)

    # Return the best hyperparameters
    return search_cv.best_params_




