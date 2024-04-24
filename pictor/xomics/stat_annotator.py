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
# ====-========================================================================-
from pictor.xomics.stat_analyzers import auto_dual_test
from pictor.xomics.stat_analyzers import single_factor_analysis
from roma import Nomear

import matplotlib.pyplot as plt



class Annotator(Nomear):

  def __init__(self, groups: list, ax: plt.Axes):
    self.groups = groups
    self.ax = ax

  def annotate(self, n_top=2):
    reports = single_factor_analysis(self.groups)

    # Get current y limits
    y_min, y_max = self.ax.get_ylim()
    h = (y_max - y_min) / 20
    lhp = 0.5
    color = 'k'
    for n, (i, j, p_val, method) in enumerate(reports):
      y = y_max + n * h
      self.ax.plot([i, i, j, j], [y + h * lhp, y + h, y + h, y + h * lhp],
                   lw=1.5, c=color, alpha=0.2)
      p_val_str = f'{p_val:.2E}' if p_val < 0.001 else f'{p_val:.3f}'
      self.ax.text((i + j) * .5, y + h, f'{method}: {p_val_str}',
                   ha='center', va='bottom', color=color)
    self.ax.set_ylim(y_min, y_max + (n + n_top) * h)


