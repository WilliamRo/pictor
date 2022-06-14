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
from typing import Optional, Union
from .plotter_base import Plotter
from pictor.objects.signals import SignalGroup
from pictor.objects.signals.scrolling import Scrolling

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



class Oscilloscope(Plotter):
  """An Oscilloscope can plot multiple signals recorded by different sensors
  simultaneously. The object should an instance of
  """

  def __init__(self, pictor=None, default_win_size=None, y_ticks=True):
    # Call parent's constructor
    super(Oscilloscope, self).__init__(self.show_signal, pictor)

    # Specific attributes
    self.scroll_buffer = {}
    self._selected_signal: Optional[Scrolling] = None

    # Settable attributes
    self.new_settable_attr('default_win_size', default_win_size,
                           int, 'Default window size')
    self.new_settable_attr('step', 0.2, float, 'Window moving step')
    self.new_settable_attr('bar', True, bool,
                           'Whether to show a location bar at the bottom')
    self.new_settable_attr('channels', '*', str,
                           'Channels to display, `all` by default')
    self.new_settable_attr('y_ticks', y_ticks, bool, 'Whether to show y-ticks')

  # region: Plot Method

  def show_signal(self, x: np.ndarray, fig: plt.Figure, i: int):
    # Clear figure
    fig.clear()

    # If x is not provided
    if x is None:
      self.show_text('No signal found', fig=fig)
      return

    # Get a Scrolling object based on input x
    s = self._get_scroll(x, i)
    self._selected_signal = s

    # Get channels [(name, x, y)]
    channels = s.get_channels(self.get('channels'))

    # Create subplots
    height_ratios = [1 for _ in channels]
    if self.get('bar'): height_ratios.append(sum(height_ratios) / 10)
    axs = fig.subplots(len(height_ratios), 1,
                       gridspec_kw={'height_ratios': height_ratios})

    # Plot signals
    for i, (name, x, y) in enumerate(channels):
      self._plot_signal(axs[i], name, x, y, x_ticks=i==len(channels)-1,
                        title=s.label if i == 0 else None)

    # Show scroll-bar if necessary
    if self.get('bar'): self._outline_bar(axs[-1], s)

  def _plot_signal(self, ax: plt.Axes, name, x, y, x_ticks=True, title=None):
    ax.plot(x, y)
    ax.set_ylabel(name, rotation=90)
    ax.set_xlim(min(x), max(x))
    ax.get_xaxis().set_visible(x_ticks)
    # Set styles
    if not self.get('y_ticks'): ax.set_yticklabels([])
    # Set title if provided
    if title is not None: ax.set_title(title)

  def _outline_bar(self, ax: plt.Axes, s: Scrolling):
    """Reference: https://matplotlib.org/stable/tutorials/intermediate/arranging_axes.html"""
    ticks = s.dominate_signal.ticks
    start_i = int(len(ticks) * s.start_position)

    # Create a rectangular patch
    rect = patches.Rectangle(
      (ticks[start_i], 0), width=s.window_size-1, height=1, edgecolor='#F66',
      linewidth=4, facecolor='none')
    # Add the patch to ax
    ax.add_patch(rect)
    # Set axis style
    ax.set_xlim(ticks[0], ticks[-1])
    ax.get_yaxis().set_visible(False)

  # endregion: Plot Method

  # region: Public Methods

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

    self.register_a_shortcut('b', lambda: self.flip('bar'),
                             description='Toggle location bar')
    self.register_a_shortcut('y', lambda: self.flip('y_ticks'),
                             description='Whether to show y-ticks')

  # endregion: Commands and Shortcuts

  # region: Private Methods

  def _get_scroll(self, x, i: int):
    if isinstance(x, Scrolling): return x
    if isinstance(x, SignalGroup):
      x.__class__ = Scrolling
      return x
    # numpy array is not hashable, so its memory address will be used instead
    key = i
    if key not in self.scroll_buffer:
      # Initialize a DigitalSignal if not found in buffer
      s = Scrolling(x)
      # Set window size if a default value is provided
      win_size = self.get('default_win_size')
      if win_size is not None: s.window_size = win_size
      self.scroll_buffer[key] = s
    return self.scroll_buffer[key]

  # endregion: Private Methods
