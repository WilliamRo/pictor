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
from roma import Nomear



class ROC(Nomear):
  """Receiver operating characteristic (ROC)."""

  @classmethod
  def calc_auc(cls, probs, targets, return_fpr_tpr=False):
    """Calculate the area under the curve (AUC) of the ROC curve.
    Only works for binary classification with targets in {0, 1}.
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(targets, probs)
    auc = auc(fpr, tpr)

    if return_fpr_tpr: return auc, fpr, tpr
    return auc


  @classmethod
  def plot_roc(cls, probs, targets):
    import matplotlib.pyplot as plt

    auc, fpr, tpr = cls.calc_auc(probs, targets, return_fpr_tpr=True)

    plt.plot([0, 1], [0, 1], c='grey', alpha=0.5, linestyle='--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUC = {auc:.3f}')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

