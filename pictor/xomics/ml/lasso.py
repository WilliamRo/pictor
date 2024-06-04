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
  DEFAULT_HP_SPACE = {'alpha': np.logspace(-6, 1, 50)}
  DEFAULT_HP_MODEL_INIT_KWARGS = {'tol': 1e-2}

  @property
  def selected_indices(self): return self.get_from_pocket('selected_indices')

  def select_features(self, omix: Omix, **kwargs):
    # (0) get settings
    verbose = kwargs.get('verbose', self.verbose)
    threshold = kwargs.get('threshold', 0.001)
    plot_path = kwargs.get('plot_path', False)
    alpha = kwargs.get('alpha', None)

    # (1) Tune hyperparameters
    if alpha is None:
      if plot_path:
        alpha = self.tune_alpha(omix, **kwargs)
        hp = {'alpha': alpha}
      else:
        hp = self.tune_hyperparameters(omix, **kwargs)
    else:
      hp = {'alpha': alpha}

    # (2) Fit model and get importance
    lasso = self.fit(omix, hp=hp, **kwargs)
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


  def tune_alpha(self, omix: Omix, alphas=DEFAULT_HP_SPACE['alpha'],
                 **kwargs):
    from sklearn.linear_model import LassoCV, lasso_path
    import matplotlib.pyplot as plt

    # (1) Get settings
    random_state = kwargs.get('random_state', None)
    n_splits = kwargs.get('n_splits', 5)
    X, y = omix.features, omix.targets

    # (2) Generate path
    lasso_cv = LassoCV(alphas=alphas, cv=n_splits, random_state=random_state)
    lasso_cv.fit(X, y)

    best_alpha = lasso_cv.alpha_
    # self.best_hp['alpha'] = best_alpha
    if not kwargs.get('plot_path'): return

    # (3) Plot path
    # vl_color = '#5b79ca'
    vl_color = '#bd3831'

    alphas, coefs, dual_gaps = lasso_path(X, y, alphas=alphas)
    log_alphas = np.log10(alphas)

    # Set plot style
    # plt.style.use('ggplot')
    fig = plt.figure(figsize=(12, 5))

    # (3.1) Plot the best alpha found by LassoCV
    ax1 = fig.add_subplot(1, 2, 1)
    for coef in coefs: ax1.plot(log_alphas, coef, lw=2)

    ax1.axvline(np.log10(best_alpha), linestyle='--', color=vl_color,
                label=rf'Best $\alpha$: {best_alpha:.4f}')
    # ax2 = plt.gca().twinx()
    # ax2.plot(log_alphas, dual_gaps, ':', color='k', label='Dual Gaps')

    ax1.set_xlabel(r'Log$_{10}(\alpha$)')
    ax1.set_ylabel('Features')
    ax1.set_title('LASSO Paths')
    ax1.grid(True)
    # plt.grid(True)
    ax1.legend()

    # (3.2)
    mse_mus = np.apply_along_axis(np.mean, 1, lasso_cv.mse_path_)
    mse_stds = np.apply_along_axis(np.std, 1, lasso_cv.mse_path_)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.errorbar(log_alphas, mse_mus, yerr=mse_stds, fmt='o',
                 mfc='r', mec='r', ecolor='#AAA', capsize=3, ms=3,)

    ax2.axvline(np.log10(best_alpha), linestyle='--', color=vl_color)

    ax2.set_xlabel(r'Log$_{10}(\alpha$)')
    ax2.set_ylabel('Mean Squared Error (MSE)')
    # ax2.grid(True)
    # ax2.set_title('LASSO Paths')

    # ax2.legend()

    plt.tight_layout()
    plt.show()

    return best_alpha

