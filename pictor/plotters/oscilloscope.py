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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Union
from .plotter_base import Plotter
from ..objects.digital_signal import DigitalSignal
from ..widgets.outline_bar import OutlineBar

import matplotlib.pyplot as plt
import numpy as np



class Oscilloscope(Plotter):

  def __init__(self, pictor=None, default_win_size=None):
    # Call parent's constructor
    super(Oscilloscope, self).__init__(self.show_signal, pictor)

    # Specific attributes
    self.signal_buffer = {}
    self._selected_signal: Optional[DigitalSignal] = None
    self._outline_bar: Optional[OutlineBar] = None

    # Settable attributes
    self.new_settable_attr('default_win_size', default_win_size,
                           int, 'Default window size')
    self.new_settable_attr('step', 0.2, float, 'Window moving step')

  # region: Plot Method

  def show_signal(self, x: np.ndarray, ax: plt.Axes, i: int):
    if x is None: return
    ds = self._get_x(x, i)
    # Make sure this signal can be accessed by other methods
    self._selected_signal = ds

    # Plot
    ax.plot(ds.x, ds.y)

    # Show window
    ax.set_xlim(*ds.xlim)
    if isinstance(self._outline_bar, OutlineBar):
      self._outline_bar.locate(*ds.window_location_pct)

  # endregion: Plot Method

  # region: Public Methods

  def link_to_outline_bar(self, outline_bar: OutlineBar):
    self._outline_bar = outline_bar

  # endregion: Public Methods

  # region: Commands and Shortcuts

  def move_window(self, direction=1, go_extreme=False):
    self._selected_signal.move_window(direction * self.get('step'), go_extreme)
    self.refresh()

  def set_win_size(self, multiplier):
    self._selected_signal.set_window_size(multiplier)
    self.refresh()

  def register_shortcuts(self):
    self.register_a_shortcut('h', lambda: self.move_window(-1),
                             description='Slide window to left')
    self.register_a_shortcut('l', lambda: self.move_window(1),
                             description='Slide window to right')
    self.register_a_shortcut('H', lambda: self.move_window(-1, True),
                             description='Slide window to left most')
    self.register_a_shortcut('L', lambda: self.move_window(1, True),
                             description='Slide window to right most')
    self.register_a_shortcut('o', lambda: self.set_win_size(2),
                             description='Double window size')
    self.register_a_shortcut('i', lambda: self.set_win_size(0.5),
                             description='Halve window size')

  # endregion: Commands and Shortcuts

  # region: Private Methods

  def _get_x(self, x: Union[np.ndarray, DigitalSignal], i: int):
    if isinstance(x, DigitalSignal): return x
    # numpy array is not hashable, so its memory address will be used instead
    key = i
    if key not in self.signal_buffer:
      # Initialize a DigitalSignal if not found in buffer
      s = DigitalSignal(x)
      # Set window size if a default value is provided
      win_size = self.get('default_win_size')
      if win_size is not None: s.window_size = win_size
      self.signal_buffer[key] = s
    return self.signal_buffer[key]

  # endregion: Private Methods
