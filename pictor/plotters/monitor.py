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
# ==-=======================================================================-===
from typing import Optional, List
from .plotter_base import Plotter
from pictor.objects.signals import SignalGroup
from pictor.objects.signals.scrolling import Scrolling

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



class Monitor(Plotter):
  """A Monitor can plot multiple records simultaneously. The object should an instance of
  """

  def __init__(self, pictor=None, window_duration=60,
               channels: str='*', y_ticks=True):
    """
    :param window_duration: uses second as unit
    """
    # Call parent's constructor
    super(Monitor, self).__init__(self.show_curves, pictor)

    # Specific attributes
    self.scroll_buffer = {}
    self._selected_signal: Optional[Scrolling] = None

    # Settable attributes
    self.new_settable_attr('default_win_duration', window_duration,
                           float, 'Default window duration')
    self.new_settable_attr('step', 0.1, float, 'Window moving step')
    self.new_settable_attr('bar', True, bool,
                           'Whether to show a location bar at the bottom')
    self.new_settable_attr('channels', channels, str,
                           'Channels to display, `all` by default')
    self.new_settable_attr('y_ticks', y_ticks, bool, 'Whether to show y-ticks')

  # region: Properties

  @property
  def channel_list(self):
    return self.get_from_pocket(
      'channel_list', default=[
        c for c, _, _ in self._selected_signal.name_tick_data_list])

  @channel_list.setter
  def channel_list(self, value):
    assert isinstance(value, (list, tuple))
    self.put_into_pocket('channel_list', value)

  # endregion: Properties

  # region: Plot Method
  
  def register_to_master(self, pictor):
    # pictor.canvas._canvas.mpl_connect('button_press_event', lambda e: print(
    #   f'({e.xdata}, {e.ydata})'))
    super(Monitor, self).register_to_master(pictor)

    # Set channel hints
    def get_hints():
      hints = ['Channel list', '-' * 12]
      hints += [f'[{i+1}] {name}' for i, name in enumerate(self.channel_list)]
      return '\n'.join(hints)
    self.pictor.command_hints['sc'] = get_hints
    self.pictor.command_hints['set_channels'] = get_hints

  def show_curves(self, x: np.ndarray, fig: plt.Figure, i: int):
    # Clear figure
    fig.clear()

    # If x is not provided
    if x is None:
      self.show_text('No signal found', fig=fig)
      return

    # Get a Scrolling object based on input x
    s = self._get_scroll(x, i)
    self._selected_signal = s

    # Create subplots
    height_ratios = [50]
    if self.get('bar'): height_ratios.append(1)
    axs = fig.subplots(
      len(height_ratios), 1, gridspec_kw={'height_ratios': height_ratios})

    # Plot signals
    ax: plt.Axes = axs[0] if len(height_ratios) > 1 else axs
    self._plot_curve(ax, s)

    # Show scroll-bar if necessary
    if self.get('bar'): self._outline_bar(axs[-1], s)

  def _plot_curve(self, ax: plt.Axes, s: Scrolling):
    """ i  y
           2  ---------
        0     -> N(=2) - i(=0) - 0.5 = 1.5
           1  ---------
        1     -> N(=2) - i(=1) - 0.5 = 0.5
           0  ---------
    """
    # Get channels [(name, x, y)]
    channels = s.get_channels(self.get('channels'))
    N = len(channels)

    margin = 0.1
    for i, (name, x, y) in enumerate(channels):
      # Plot normalized y
      y -= min(y)
      y = y / max(y) * (1.0 - 2 * margin) + margin
      y += N - 1 - i
      ax.plot(x, y, color='black', linewidth=1)

      # Set xlim
      if i == 0: ax.set_xlim(x[0], x[-1])

    # Set y_ticks
    ax.set_yticks([N - i - 0.5 for i in range(N)])
    ax.set_yticklabels([name for name, _, _ in channels])

    # Set styles
    ax.set_ylim(0, N)
    ax.grid(color='#E03', alpha=0.4)

    ax.set_title(s.label)

  def _outline_bar(self, ax: plt.Axes, s: Scrolling):
    """Reference: https://matplotlib.org/stable/tutorials/intermediate/arranging_axes.html"""
    ticks = s.dominate_signal.ticks
    start_i = int(len(ticks) * s.start_position)

    # Create a rectangular patch
    width = s.total_duration * s.window_size
    rect = patches.Rectangle(
      (ticks[start_i], 0), width=width, height=1, edgecolor='#F66',
      linewidth=0, facecolor='#F66')
    # Add the patch to ax
    ax.add_patch(rect)
    # Set axis style
    ax.set_xlim(ticks[0], ticks[-1])
    ax.get_yaxis().set_visible(False)

  # endregion: Plot Method

  # region: Public Methods

  def goto(self, time: float):
    ss = self._selected_signal
    ds = ss.dominate_signal
    self.goto_position((time - ds.ticks[0]) / ss.total_duration)

  # endregion: Public Methods

  # region: Commands and Shortcuts

  def move_window(self, direction=1, go_extreme=False):
    self._selected_signal.move_window(direction * self.get('step'), go_extreme)
    self.refresh()

  def goto_position(self, position: int):
    self._selected_signal.start_position = position
    self.refresh()

  def set_win_size(self, multiplier: int):
    self._selected_signal.set_window_size(multiplier)
    self.refresh()

  def set_win_duration(self, duration: float):
    assert duration > 0
    self._selected_signal.set_window_duration(duration)
    self.refresh()
  sd = set_win_duration

  def set_channels(self, channels: str):
    all_channels = self.channel_list
    if channels == '*': channels = all_channels
    else: channels = [all_channels[int(id) - 1] for id in channels.split(',')]
    self.set('channels', ','.join(channels))
  sc = set_channels

  def register_shortcuts(self):
    self.register_a_shortcut('h', lambda: self.move_window(-1),
                             description='Slide window to left')
    self.register_a_shortcut('l', lambda: self.move_window(1),
                             description='Slide window to right')
    self.register_a_shortcut('H', lambda: self.move_window(-1, True),
                             description='Slide window to left most')
    self.register_a_shortcut('L', lambda: self.move_window(1, True),
                             description='Slide window to right most')
    self.register_a_shortcut('M', lambda: self.goto_position(0.5),
                             description='Slide window to middle')
    self.register_a_shortcut('o', lambda: self.set_win_size(2.0),
                             description='Double window size')
    self.register_a_shortcut('i', lambda: self.set_win_size(0.5),
                             description='Halve window size')

    self.register_a_shortcut('b', lambda: self.flip('bar'),
                             description='Toggle location bar')
    self.register_a_shortcut('y', lambda: self.flip('y_ticks'),
                             description='Whether to show y-ticks')

  # endregion: Commands and Shortcuts

  # region: Private Methods

  def _get_scroll(self, x, i: int) -> Scrolling:
    # Since numpy array is not hashable, its id will be used as key
    if i in self.scroll_buffer: return self.scroll_buffer[i]

    # If x not in buffer
    if isinstance(x, SignalGroup):
      # Convert a SignalGroup to a Scrolling safely if necessary
      if not isinstance(x, Scrolling): x.__class__ = Scrolling
    elif isinstance(x, np.ndarray): x = Scrolling(x)
    else: raise TypeError(f'!! Illegal type ({type(x)}) for scrolling')

    # Set default win_size if provided
    x.set_window_duration(self.get('default_win_duration'))

    # Put x in buffer and return
    self.scroll_buffer[i] = x
    return x

  # endregion: Private Methods
