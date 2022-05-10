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
# ===-==================================================================-=======
from .widget_base import WidgetBase

import tkinter as tk



class OutlineBar(WidgetBase):

  def __init__(self, pictor, height=50):
    from ..pictor import Pictor
    self.pictor: Pictor = pictor

    self.canvas = tk.Canvas(master=pictor, height=height)

    # Call parent's constructor
    super(OutlineBar, self).__init__(tk_widget=self.canvas)

  # region: Widget Style

  def _set_style(self):
    self.canvas.configure(background='#F6F6F6')

  # endregion: Widget Style

  # region: Public Methods

  def locate(self, a, b):
    assert 0 <= a < b <= 1
    # Get canvas size (tricky, need to be refactored)
    H = self.canvas.winfo_reqheight()
    W = self.pictor.canvas._tk_widget.winfo_reqwidth() - 10
    # Define margin
    m = 5

    x_min = max(W * a, m)
    x_max = min(W * b, W)
    self.canvas.delete('all')
    self.canvas.create_rectangle(
      x_min, m, x_max, H - m, outline='#FA8', fill='#FFF', width=2)

  # endregion: Public Methods

