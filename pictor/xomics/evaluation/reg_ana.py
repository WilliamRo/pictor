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
# ===-=======================================================================-==
import matplotlib.pyplot as plt
import numpy as np

from roma import Nomear
from scipy import stats



class RegressionAnalysis(Nomear):
  """Regression analysis."""

  def __init__(self, true_y, pred_y):
    self.true_y = true_y
    self.pred_y = pred_y

  @Nomear.property()
  def mae(self):
    return np.mean(np.abs(self.true_y - self.pred_y))

  @Nomear.property()
  def r(self):
    _r, _pval = stats.pearsonr(self.true_y, self.pred_y)
    return _r

  @Nomear.property()
  def r_CI95(self):
    n = len(self.true_y)
    r_z = np.arctanh(self.r)  # Fisher transformation
    se = 1 / np.sqrt(n - 3)
    z = stats.norm.ppf(0.975)
    ci_lower = np.tanh(r_z - z * se)
    ci_upper = np.tanh(r_z + z * se)
    return ci_lower, ci_upper

  def plot_scatter(self, ax: plt.Axes=None, title=None, **kwargs):
    import matplotlib.pyplot as plt

    plt_show = ax is None

    # Create figure and axis if not provided
    if ax is None:
      fig = plt.figure(**kwargs)
      ax: plt.Axes = fig.add_subplot(111)

    # Plot ideal line
    min_y, max_y = min(self.true_y), max(self.true_y)
    ax.plot([min_y, max_y], [min_y, max_y], 'r--', label='Identity line')

    # Plot data
    label = f'MAE = {self.mae:.2f}'
    lo, hi = self.r_CI95
    label +=f'\nr = {self.r:.2f} (CI95% = [{lo:.2f}, {hi:.2f}])'
    ax.plot(self.true_y, self.pred_y, 'o', label=label, alpha=0.8)

    # Set x, y labels
    xlabel = kwargs.get('xlabel', 'True Values')
    ylabel = kwargs.get('ylabel', 'Predicted Values')
    ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)

    # Set title if provided
    if title is not None: ax.set_title(title)

    ax.legend(fontsize=kwargs.get('fontsize', 10))
    ax.grid(True)

    if plt_show:
      plt.tight_layout()
      plt.show()
