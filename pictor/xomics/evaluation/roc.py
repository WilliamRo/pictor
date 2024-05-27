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

  def plot_roc(self, ax: plt.Axes=None, **kwargs):
    import matplotlib.pyplot as plt

    fpr, tpr, _ = self.roc_curve
    auc = self.auc

    plt_show = ax is None

    if ax is None:
      fig = plt.figure(**kwargs)
      ax: plt.Axes = fig.add_subplot(111)

    ax.plot([0, 1], [0, 1], c='grey', alpha=0.5, linestyle='--')
    ax.plot(fpr, tpr)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'AUC = {auc:.3f}')

    ax.grid(True)

    if plt_show:
      plt.tight_layout()
      plt.show()

