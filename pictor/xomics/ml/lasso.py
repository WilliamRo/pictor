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
from sklearn.linear_model import Lasso as SKLasso

import numpy as np



class Lasso(MLEngine):
  SK_CLASS = SKLasso
  DEFAULT_HP_SPACE = {'alpha': np.logspace(-6, 1, 20)}
  DEFAULT_HP_MODEL_INIT_KWARGS = {'tol': 1e-2}

  @property
  def selected_indices(self): return self.get_from_pocket('selected_indices')

  def select_features(self, omix: Omix, **kwargs):
    # (0) get settings
    verbose = kwargs.get('verbose', self.verbose)
    threshold = kwargs.get('threshold', 0.001)

    # (1) Tune hyperparameters
    self.tune_hyperparameters(omix)

    # (2) Fit model and get importance
    lasso = self.fit(omix, **kwargs)
    importance = np.abs(lasso.coef_)

    if verbose > 1: self.plot_importance(importance, omix.feature_labels)

    # (3) Select features
    indices = np.where(importance > threshold)[0]
    self.put_into_pocket('selected_indices', indices, local=True)
    selected_features = omix.features[:, indices]
    labels = np.array(omix.feature_labels)[indices]

    return omix.duplicate(features=selected_features, feature_labels=labels)


  @staticmethod
  def plot_importance(importance, labels):
    import matplotlib.pyplot as plt

    MAX_LEN = 20
    labels = [f'{label[:MAX_LEN]}' for label in labels]

    # Note that plt.bar will merge the same labels
    plt.bar(labels, importance)
    plt.xticks(rotation=45)
    plt.grid()
    plt.title("Lasso Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.ylim(0, 1.2 * max(importance))
    plt.tight_layout()
    plt.show()
