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
# ====-==================================================================-======
from .plotter_base import Plotter

import matplotlib.pyplot as plt



class Prompter(Plotter):

  def __init__(self, obj_as_text=True, text=None, pictor=None):
    # Call parent's constructor
    if obj_as_text: func = lambda ax, obj: self.show_text(ax, obj)
    else: func = self.show_text
    super(Prompter, self).__init__(func, pictor)

    # Specific attributes
    self.text = text


  def show_text(self, ax: plt.Axes, text):
    ax.cla()
    ax.text(0.5, 0.5, text, ha='center', va='center')
    ax.set_axis_off()


  def set_text(self, val: str):
    self.text = val
    self.refresh()

