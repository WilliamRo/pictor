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
from pictor.xomics.ml.ml_engine import MLEngine, Omix
from roma import console
from sklearn.linear_model import Lasso

import numpy as np



def select_features(omix: Omix, **kwargs):
    """Select features using Lasso regression"""
    # Tune hyperparameters
    n_splits = kwargs.get('n_splits', 5)
    strategy = kwargs.get('strategy', 'grid')
    random_state = kwargs.get('random_state', None)
    hp_space = kwargs.get('hp_space', {'alpha': np.logspace(-6, 1, 20)})
    verbose = kwargs.get('verbose', 0)
    tol = kwargs.get('tol', 1e-2)

    if verbose == 0:
      import warnings
      warnings.filterwarnings('ignore')

    ml = MLEngine(omix, random_state=random_state)
    hp = ml.tune_hyperparameters(
      Lasso(tol=tol), hp_space, n_splits=n_splits, strategy=strategy)

    # Fit data using best hyperparameters
    lasso = Lasso(**hp)
    lasso.fit(omix.features, omix.targets)
    importance = np.abs(lasso.coef_)

    # Show status if necessary
    if verbose > 0: console.show_status(f'Best hyperparameters: {hp}')

    # Plot feature importance if required
    if verbose > 1:
      import matplotlib.pyplot as plt

      MAX_LEN = 20
      labels = [f'{label[:MAX_LEN]}' for label in omix.feature_labels]
      # plt.bar will merge the same labels
      plt.bar(labels, importance)
      plt.xticks(rotation=45)
      plt.grid()
      plt.title("Lasso Feature Importance")
      plt.xlabel("Features")
      plt.ylabel("Importance")
      plt.ylim(0, 1.2 * max(importance))
      plt.tight_layout()
      plt.show()

    # Select features
    threshold = kwargs.get('threshold', 0.001)
    indices = np.where(importance > threshold)[0]
    selected_features = omix.features[:, indices]
    labels = np.array(omix.feature_labels)[indices]

    return omix.duplicate(features=selected_features, feature_labels=labels)


