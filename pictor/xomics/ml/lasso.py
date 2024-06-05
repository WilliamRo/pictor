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
      alpha = self.tune_alpha(omix, **kwargs)
      hp = {'alpha': alpha}

      # if plot_path:
      #   alpha = self.tune_alpha(omix, **kwargs)
      #   hp = {'alpha': alpha}
      # else:
      #   hp = self.tune_hyperparameters(omix, **kwargs)
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
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt

    # (1) Get settings
    random_state = kwargs.get('random_state', None)
    n_splits = kwargs.get('n_splits', 5)
    n_repeats = kwargs.get('lasso_repeats', 1)
    if random_state is not None: n_repeats = 1

    verbose = kwargs.get('verbose', 0)
    log_alphas = np.log10(alphas)

    prompt = '[LASSO]'

    # (2) Generate path
    X, y = omix.features, omix.targets

    lasso_cv_list = []
    if verbose: console.show_status(
      f'Tuning alpha ({n_repeats} repeats) ...', prompt=prompt)
    for _ in range(n_repeats):
      kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
      lasso_cv = LassoCV(alphas=alphas, cv=kf, random_state=random_state)
      lasso_cv.fit(X, y)
      lasso_cv_list.append(lasso_cv)

    # (2.1) Select the best alpha
    mse_paths = [lasso_cv.mse_path_ for lasso_cv in lasso_cv_list]
    merged_mse_path = np.concatenate(mse_paths, axis=1)
    mean_path = np.mean(merged_mse_path, axis=1)
    # !! Note here, the order of mean_path has been reversed
    mean_path = mean_path[::-1]
    best_mmse = np.min(mean_path)
    best_alpha = alphas[np.argmin(mean_path)]

    if verbose: console.show_status(
      f'Best alpha = {best_alpha:.4f}. '
      f'Best mean(MSE) = {best_mmse:.4f}', prompt=prompt)

    # (2.2) Sanity check / return
    if n_repeats == 1: assert best_alpha == lasso_cv.alpha_

    if not kwargs.get('plot_path'): return best_alpha

    # (3) Plot path
    # vl_color = '#5b79ca'
    vl_color = '#bd3831'

    # !! Note here, alpha_p is a reverse of alphas
    alphas_p, coefs, dual_gaps = lasso_path(X, y, alphas=alphas)
    log_alphas_p = np.log10(alphas_p)

    # Set plot style
    # plt.style.use('ggplot')
    fig = plt.figure(figsize=(12, 5))

    # (3.1) Plot the best alpha found by LassoCV
    ax1 = fig.add_subplot(1, 2, 1)
    for coef in coefs: ax1.plot(log_alphas_p, coef, lw=2)

    ax1.axvline(np.log10(best_alpha), linestyle='--', color=vl_color,
                label=rf'Best $\alpha$: {best_alpha:.4f}')

    ax1.set_xlabel(r'Log$_{10}(\alpha$)')
    ax1.set_ylabel('Features')
    ax1.set_title('LASSO Paths')
    ax1.grid(True)
    ax1.legend()

    # (3.2)
    mse_mus = mean_path
    mse_stds = np.std(merged_mse_path, axis=1)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.errorbar(log_alphas, mse_mus, yerr=mse_stds, fmt='o',
                 mfc='r', mec='r', ecolor='#AAA', capsize=3, ms=3,)

    ax2.axvline(np.log10(best_alpha), linestyle='--', color=vl_color,
                label=rf'Best mean(MSE): {best_mmse:.4f}')

    ax2.set_xlabel(r'Log$_{10}(\alpha$)')
    ax2.set_ylabel('Mean Squared Error (MSE)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return best_alpha

