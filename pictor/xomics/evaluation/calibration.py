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
# ===-===============================================================-==========
from roma import Nomear
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np



class Calibrator(Nomear):

  def __init__(self, prob_dict: OrderedDict, true_dict: OrderedDict, **kwargs):
    self.prob_dict = prob_dict
    self.true_dict = true_dict

    self.configs = kwargs


  @Nomear.property()
  def true_prob_dict(self) -> OrderedDict:
    od = OrderedDict()
    for key, prob in self.prob_dict.items():
      # Sanity check
      assert key in self.true_dict
      assert isinstance(prob, (tuple, list))

      targ = self.true_dict[key]
      assert isinstance(targ, (tuple, list)) and len(targ) == len(prob)

      # Users, do not use this unless being confident
      if 'k' in self.configs:
        k = self.configs['k']
        prob, targ = prob[:k], targ[:k]
        print(f'[Calibrator Developer] Repeat number = {len(prob)}.')

      prob = np.concatenate(prob)
      y_true = np.concatenate(targ)

      od[key] = (y_true, prob)

    return od


  def plot_calibration_curve(self, n_bins=10, ax: plt.Axes=None, **kwargs):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    # (0) Get settings, init ax
    plt_show = ax is None
    if plt_show:
      fig = plt.figure(**kwargs)
      ax: plt.Axes = fig.add_subplot(111)

    # (1) Plot curves
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

    for key, (y_true, prob) in self.true_prob_dict.items():
      b_score = brier_score_loss(y_true, prob)
      true_pos, pred_pos = calibration_curve(y_true, prob, n_bins=n_bins)
      label = f'{key}, Brier Score: {b_score:.3f}'
      ax.plot(pred_pos, true_pos, marker='o', label=label, alpha=0.8)

    # (2) Set styles
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # (2.2) MISC
    ax.set_title('Probability Calibration Curve')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Probability')
    ax.legend()

    # (-1) plt show if required
    if plt_show:
      plt.tight_layout()
      plt.show()
