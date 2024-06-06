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

from . import delong
from roma import Nomear



class ROC(Nomear):
  """Receiver operating characteristic (ROC)."""

  def __init__(self, probs, targets):
    self.probs = probs
    self.targets = targets

  @Nomear.property()
  def roc_curve(self):
    from sklearn.metrics import roc_curve
    return roc_curve(self.targets, self.probs)

  @Nomear.property()
  def auc(self):
    from sklearn.metrics import auc
    fpr, tpr, _ = self.roc_curve
    return auc(fpr, tpr)

  def plot_roc(self, ax: plt.Axes=None, label=None, **kwargs):
    import matplotlib.pyplot as plt

    fpr, tpr, _ = self.roc_curve
    auc = self.auc

    plt_show = ax is None

    if ax is None:
      fig = plt.figure(**kwargs)
      ax: plt.Axes = fig.add_subplot(111)

    ax.plot([0, 1], [0, 1], c='grey', alpha=0.5, linestyle='--')

    l, h = self.calc_CI()
    legend = f'AUC={auc:.3f}, CI(95%)=[{l:.3f}, {h:.3f}]'

    ax.plot(fpr, tpr, label=legend)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    if label is not None: ax.set_title(label)

    ax.legend(fontsize=kwargs.get('fontsize', 10))
    ax.grid(True)

    if plt_show:
      plt.tight_layout()
      plt.show()

  def calc_CI(self, alpha=0.95):
    import scipy.stats as st

    auc_0, delong_cov = delong.delong_roc_variance(self.targets, self.probs)
    q1, q2 = (1 - alpha) / 2, (1 + alpha) / 2
    l, h = st.norm.ppf([q1, q2], loc=auc_0, scale=np.sqrt(delong_cov))
    return max(0, l), min(1, h)

