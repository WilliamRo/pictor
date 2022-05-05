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
# ====-=============================================================-===========
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from roma import Easel

from .widget_base import WidgetBase

import matplotlib.pyplot as plt



class Canvas(WidgetBase):

  def __init__(self, master: Easel, figure_size=(5, 5)):
    self.master = master
    self.figure = plt.Figure(figsize=figure_size)
    self._canvas = FigureCanvasTkAgg(self.figure, master=self.master.master)

    # Call parent's constructor
    super(Canvas, self).__init__(self._canvas.get_tk_widget())

  # region: Private Methods

  # endregion: Private Methods

  # region: Abstract Methods

  def refresh(self):
    from ..pictor import Pictor
    # Get active plotter
    plotter = self.master.get_element(Pictor.Keys.PLOTTERS)
    if plotter is None: plotter = lambda: self.show_message('No plotter found')

    # Call plotter
    plotter()

    # Flush buffer
    self._canvas.draw()

  # endregion: Abstract Methods

  # region: Builtin Plotters

  def show_message(self, text):
    ax: plt.Axes = self.figure.add_subplot(111)
    ax.cla()
    ax.text(0.5, 0.5, text, ha='center', va='center')
    ax.set_axis_off()

  # endregion: Builtin Plotters
