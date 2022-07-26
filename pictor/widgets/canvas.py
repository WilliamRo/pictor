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
from mpl_toolkits.mplot3d import Axes3D

from .widget_base import WidgetBase

import matplotlib.pyplot as plt



class Canvas(WidgetBase):

  class Keys:
    AXES2D = 'AxEs2d'
    AXES3D = 'AxEs3d'
    VIEW_ANGLE_3D = 'ViEw_AnGlE_3d'

  def __init__(self, pictor, figure_size=(5, 5)):
    from ..pictor import Pictor
    self.pictor: Pictor = pictor
    self.figure = plt.Figure(figsize=figure_size, dpi=100)
    self._canvas = FigureCanvasTkAgg(self.figure, master=self.pictor)

    # Call parent's constructor
    super(Canvas, self).__init__(self._canvas.get_tk_widget())

  # region: Properties

  @WidgetBase.property(key=Keys.AXES2D)
  def axes2D(self) -> plt.Axes:
    return self.figure.add_subplot(111)

  @WidgetBase.property(key=Keys.AXES3D)
  def axes3D(self) -> Axes3D:
    return self.figure.add_subplot(111, projection='3d')

  @property
  def view_angle(self):
    return self.get_from_pocket(self.Keys.VIEW_ANGLE_3D, default=(None, None))

  @view_angle.setter
  def view_angle(self, val):
    assert isinstance(val, (list, tuple)) and len(val) == 2
    self.put_into_pocket(self.Keys.VIEW_ANGLE_3D, val, exclusive=False)

  @WidgetBase.property()
  def default_plotter(self):
    from ..plotters.prompter import Prompter
    return Prompter(obj_as_text=False, text='No plotter found',
                    pictor=self.pictor)

  # endregion: Properties

  # region: Private Methods

  def _clear(self):
    # Clear 2D axes if exists
    if self.in_pocket(self.Keys.AXES2D):
      self.get_from_pocket(self.Keys.AXES2D, put_back=False)

    # Clear 3D axes if exists
    if self.in_pocket(self.Keys.AXES3D):
      axes3d = self.get_from_pocket(self.Keys.AXES3D, put_back=False)
      self.view_angle = (axes3d.elev, axes3d.azim)

    # Clear figure
    self.figure.clear()

  # endregion: Private Methods

  # region: Abstract Methods

  def refresh(self, wait_for_idle=False):
    # Clear figure
    self._clear()

    # Call active plotter
    self.pictor.active_plotter()

    # Tight layout and refresh
    self.figure.tight_layout()

    if wait_for_idle: self._canvas.draw_idle()
    else: self._canvas.draw()

  # endregion: Abstract Methods

  # region: Public Methods

  def move_view_angle(self, d_elev: float, d_azim: float):
    if self.view_angle[0] is None: return
    self.view_angle = (self.view_angle[0] + d_elev, self.view_angle[1] + d_azim)

  # endregion: Public Methods
