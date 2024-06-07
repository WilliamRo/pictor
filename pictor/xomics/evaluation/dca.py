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
# ===-=====================================================================-====
from collections import OrderedDict
from roma import Nomear

import matplotlib.pyplot as plt
import numpy as np



class DCA(Nomear):
  """Decision Curve Analysis (DCA) for evaluating prediction models."""

  def __init__(self, prob_dict: OrderedDict, targets, nPT=201):
    self.prob_dict = prob_dict
    self.targets = targets
    self.nPT = nPT

    self._check_prob_dict()


  @Nomear.property()
  def PTs(self): return np.linspace(0, 1, self.nPT)[:-1]


  # region: Private Methods

  def _check_prob_dict(self):
    for key in self.prob_dict.keys():
      probs = self.prob_dict[key]
      if np.min(probs) >= 0 and np.max(probs) <= 1: continue
      probs = probs - np.min(probs)
      probs = probs / np.max(probs)
      self.prob_dict[key] = probs

  # endregion: Private Methods

  # region: Benefit Calculation

  def calc_net_benefit(self, prob):
    from sklearn.metrics import confusion_matrix

    n = len(self.targets)

    net_benefits = []
    for PT in self.PTs:
      pred = prob > PT
      tn, fp, fn, tp = confusion_matrix(self.targets, pred).ravel()
      net_benefit = tp / n - (fp / n) * (PT / (1 - PT))
      net_benefits.append(net_benefit)

    return net_benefits

  @staticmethod
  def _binary_clf_curve(y_true, y_prob, pos_label=1):
    from sklearn.metrics._ranking import _binary_clf_curve


    from sklearn.utils.extmath import stable_cumsum

    y_true = np.ravel(y_true)
    y_prob = np.ravel(y_prob)
    y_true = y_true == pos_label

    desc_prob_indices = np.argsort(y_prob, kind="mergesort")[::-1]
    y_prob = y_prob[desc_prob_indices]
    y_true = y_true[desc_prob_indices]

    distinct_value_indices = np.where(np.diff(y_prob))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_prob[threshold_idxs]

    return fps, tps, thresholds

  def calc_net_benefit_fast(self, prob):
    """This method is modified from `https://blog.csdn.net/qq_48321729/article/details/123241746`
    """
    # fps, tps, thresholds = self._binary_clf_curve(self.targets, prob)

    # tps = np.r_[0, tps]
    # fps = np.r_[0, fps]
    # thresholds = np.r_[max(thresholds[0], 1 - 1e-10), thresholds]

    from sklearn.metrics._ranking import _binary_clf_curve

    fps, tps, thresholds = _binary_clf_curve(self.targets, prob, pos_label=1)

    sort_indices = np.argsort(thresholds, kind="mergesort")
    thresholds = thresholds[sort_indices]
    tps = tps[sort_indices]
    fps = fps[sort_indices]

    n = len(self.targets)
    binids = np.searchsorted(thresholds[:-1], self.PTs)
    net_benefits = tps[binids] / n - (fps[binids] / n) * (
        self.PTs / (1 - self.PTs))

    return net_benefits

  # endregion: Benefit Calculation

  # region: Plot Methods

  def plot_dca(self, ax: plt.Axes=None, **kwargs):
    import matplotlib.pyplot as plt

    # (0) Get settings, init ax
    max_len = kwargs.get('max_len', 12)

    plt_show = ax is None
    if plt_show:
      fig = plt.figure(**kwargs)
      ax: plt.Axes = fig.add_subplot(111)
    highlight = len(self.prob_dict) == 1

    # (1) Plot curves
    # (1.1) Plot None and All
    ax.plot((0, 1), (0, 0), color='black', linestyle='--', lw=2, label='None')
    benefit_all=self.calc_net_benefit_fast(
    # benefit_all = self.calc_net_benefit(
        np.ones_like(self.targets, dtype=float))
    ax.plot(self.PTs, benefit_all, color='black', lw=2, label='All')

    # (1.2) Plot model benefits
    benefit_curves = []
    for label, prob in self.prob_dict.items():
      benefit = self.calc_net_benefit_fast(prob)
      # benefit = self.calc_net_benefit(prob)
      benefit_curves.append(benefit)
      if len(label) > max_len: label = label[:max_len] + '...'

      color = 'crimson' if highlight else None
      ax.plot(self.PTs, benefit, label=label, lw=2, color=color, alpha=0.8)

    # (2) Set styles
    # (2.1) Highlight better parts if only one model is displayed
    if highlight:
      upper = np.maximum(benefit_all, 0)
      lower = np.maximum(benefit_curves[0], upper)
      ax.fill_between(self.PTs, lower, upper, color='crimson', alpha=0.2)

    # (2.2) Set limits
    ax.set_xlim(0, 1)
    m = 0.1
    cb = np.concatenate(benefit_curves).ravel()
    ax.set_ylim(cb.min() - m, cb.max() + m)

    # (2.3) MISC
    ax.set_title('Decision Curves')
    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.grid('major')
    ax.legend()

    # (-1) plt show if required
    if plt_show:
      plt.tight_layout()
      plt.show()

  # endregion: Plot Methods


