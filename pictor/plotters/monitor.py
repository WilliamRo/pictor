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
from roma import console
from roma.spqr.arguments import Arguments

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



class Monitor(Plotter):
  """A Monitor can plot multiple records simultaneously. Each object should an
  instance of SignalGroup, or at least an numpy array.
  """

  def __init__(self, pictor=None, window_duration=60, channels: str='*'):
    """
    :param window_duration: uses second as unit
    """
    # Call parent's constructor
    super(Monitor, self).__init__(self.show_curves, pictor)

    # Specific attributes
    self.scroll_buffer = {}
    self._selected_signal: Optional[Scrolling] = None

    self._annotations_to_show = []

    # Settable attributes
    self.new_settable_attr('default_win_duration', window_duration,
                           float, 'Default window duration')
    self.new_settable_attr('step', 0.1, float, 'Window moving step')
    self.new_settable_attr('bar', True, bool,
                           'Whether to show a location bar at the bottom')
    self.new_settable_attr('channels', channels, str,
                           'Channels to display, `all` by default')
    self.new_settable_attr('max_ticks', 10000, int, 'Maximum ticks to plot')
    self.new_settable_attr('smart_scale', True, bool,
                           'Whether to use smart scale ')
    self.new_settable_attr('xi', 0.1, float, 'Margin for smart scale')
    self.new_settable_attr('hl', 0, int, 'Highlighted channel id')
    self.new_settable_attr('anno_legend', None, bool,
                           'Option to show legends of annotations')

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

  @property
  def highlighed_channel(self):
    id = int(self.get('hl'))
    return id

  @property
  def displayed_channels(self):
    channel_attr: str = self.get('channels')
    return self.channel_list if channel_attr == '*' else channel_attr.split(',')

  @property
  def hidden_channels(self):
    return [c for c in self.channel_list if c not in self.displayed_channels]

  # endregion: Properties

  # region: Plot Method
  
  def register_to_master(self, pictor):
    # pictor.canvas._canvas.mpl_connect('button_press_event', lambda e: print(
    #   f'({e.xdata}, {e.ydata})'))
    super(Monitor, self).register_to_master(pictor)

    # Set channel hints
    def get_channel_hints(for_add_channels=False):
      hints = ['Channel list', '-' * 12]
      channels = self.hidden_channels if for_add_channels else self.channel_list
      hints += [f'[{i+1}] {name}' for i, name in enumerate(channels)]
      return '\n'.join(hints)
    self.command_hints['sc'] = get_channel_hints
    self.command_hints['set_channels'] = get_channel_hints
    self.command_hints['ac'] = lambda: get_channel_hints(True)
    self.command_hints['add_channels'] = lambda: get_channel_hints(True)

    # Set annotation hints
    def get_anno_hints():
      hints = ['Annotations', '-' * 11]
      hints += [f'[{i+1}] {k}'
                for i, k in enumerate(self._selected_signal.annotations.keys())]
      return '\n'.join(hints)
    self.command_hints['ta'] = get_anno_hints
    self.command_hints['toggle_annotation'] = get_anno_hints

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

    # Plot annotations
    self._plot_annotation(ax, s)

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
    # Get settings
    smart_scale = self.get('smart_scale')
    hl_id = self.get('hl')

    # Get channels [(name, x, y)]
    channels = s.get_channels(self.get('channels'),
                              max_ticks=self.get('max_ticks'))
    N = len(channels)

    margin = 0.05
    for i, (name, x, y) in enumerate(channels):
      # Normalized y before plot
      if not smart_scale:
        y = y - min(y)
        y = y / max(y) * (1.0 - 2 * margin) + margin
      else:
        xi = self.get('xi')
        mi = s.get_channel_percentile(name, xi)
        ma = s.get_channel_percentile(name, 100 - xi)
        y = y - mi
        y = y / (ma - mi) * (1.0 - 2 * margin) + margin

      y = y + N - 1 - i
      # Plot normalized y
      color, zorder = 'black', 10
      if 0 < hl_id != i + 1: color, zorder = '#AAA', None
      ax.plot(x, y, color=color, linewidth=1, zorder=zorder)

    # Set xlim (make sure display interval \in data interval)
    tick_list = [x for _, x, _ in channels]
    ax.set_xlim(max([x[0] for x in tick_list]),
                min([x[-1] for x in tick_list]))

    # Set y_ticks
    ax.set_yticks([N - i - 0.5 for i in range(N)])
    ax.set_yticklabels([name for name, _, _ in channels])

    # Highlight label if necessary
    if hl_id > 0:
      for i, label in enumerate(ax.get_yticklabels()):
        label.set_color('black' if i + 1 == hl_id else 'grey')

    # Set styles
    ax.set_ylim(0, N)
    ax.grid(color='#E03', alpha=0.4)

    tail = f' (xi={self.get("xi")})' if smart_scale else ''
    ax.set_title(s.label + tail)

  def _plot_annotation(self, ax: plt.Axes, s: Scrolling):
    axes_dict, kwargs = {}, {}
    legend_handles = []

    for i, anno_str in enumerate(self._annotations_to_show):
      anno_config = Arguments.parse(anno_str)
      key: str = anno_config.func_name
      if key.lower() in ('sleep_stage', 'stage'):
        plot_method = self._plot_stage
        kwargs['index'] = i
      else: raise KeyError(f'!! Unknown annotation key `{key}`')

      # Try to fetch package
      start_time, end_time = ax.get_xlim()
      package = s.get_annotation(anno_str, start_time, end_time)
      if package is None: continue

      # Get results
      right_ax, line = plot_method(
        ax, axes_dict.get(key, None), package, anno_config, **kwargs)
      axes_dict[key] = right_ax
      # Set label to line
      label = anno_config.arg_list[0]
      line.set_label(label)
      legend_handles.append(line)

    # Show legend if necessary
    if len(legend_handles) > 1 or self.get('anno_legend'):
      ax.legend(handles=legend_handles, framealpha=1.0).set_zorder(99)

  def _plot_stage(self, left_ax: plt.Axes, right_ax: plt.Axes,
                  package, config: Arguments, index):
    ticks, values, labels = package

    # Determine color
    color = config.arg_dict.get('color', None)
    if color is None:
      colors = ['#4281f5', '#FF0000', '#FAC205', '#15B01A', '#000000']
      color = colors[index % len(colors)]

    # Determine other plot settings
    duration  = self._selected_signal.window_duration
    if duration < 1000: width = 16
    elif duration < 2000: width = 8
    elif duration < 4000: width = 4
    else: width = 2

    if duration < 2000: alpha = 0.3
    else: alpha = 0.6

    # Get right axes
    should_init_right_ax = right_ax is None
    if should_init_right_ax: right_ax = left_ax.twinx()

    # Plot
    line, = right_ax.plot(
      ticks, values, color=color, zorder=999, alpha=alpha, linewidth=width)

    # Set right axes if necessary
    if should_init_right_ax:
      right_ax.tick_params(axis='y', labelcolor=color)
      right_ax.set_yticks(np.arange(len(labels)))
      right_ax.set_yticklabels(labels)

      margin = 0.2
      right_ax.set_ylim(-margin, len(labels) - 1 + margin)

      right_ax.invert_yaxis()

    return right_ax, line


  def _plot_stage_(self, ax: plt.Axes, s: Scrolling):
    # Get annotation
    start_time, end_time = ax.get_xlim()
    package = s.get_annotation(self.get('annotation'), start_time, end_time)
    if package is None: return
    ticks, values, labels = package

    # Create a twin axis
    color = '#4281f5'
    ax: plt.Axes = ax.twinx()

    duration  = self._selected_signal.window_duration
    if duration < 1000: width = 16
    elif duration < 2000: width = 8
    elif duration < 4000: width = 4
    else: width = 2

    if duration < 2000: alpha = 0.3
    else: alpha = 0.6

    ax.plot(ticks, values, color=color, zorder=999, alpha=alpha,
            linewidth=width)

    # Customize y-axis
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    margin = 0.2
    ax.set_ylim(-margin, len(labels) - 1 + margin)

    ax.invert_yaxis()

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

  def toggle_annotation(self, anno_type: str = 'stage',
                        anno_label: str = 'Ground-Truth',
                        force_on=False, auto_refresh=True):
    """Show or hide specified annotations. Note that 'stage Ground-Truth' is
    the default annotation key of ground-truth stage labels in SignalGroups.
    """
    # Consider auto hints selection
    if anno_type in ('-', 'none'):
      self._annotations_to_show = []
    elif anno_type[0].isdigit():
      indices = [int(n) - 1 for n in anno_type.split(',')]
      all_anno_keys = list(self._selected_signal.annotations.keys())
      anno_keys = [all_anno_keys[i] for i in indices]
      self._annotations_to_show = anno_keys
    else:
      anno_key = f'{anno_type} {anno_label}'
      if anno_key in self._annotations_to_show:
        if not force_on: self._annotations_to_show.remove(anno_key)
      else:
        # Check anno_key before appending
        if anno_key not in self._selected_signal.annotations:
          raise KeyError(f'!! `{anno_key}` not found in Annotations')

        self._annotations_to_show.append(anno_key)

    # List all annotations to be displayed
    console.show_info('Annotations to show:')
    for i, key in enumerate(self._annotations_to_show):
      console.supplement(f'[{i+1}] {key}', level=2)

    # Refresh if required
    if auto_refresh: self.refresh()
  ta = toggle_annotation

  def anit(self, fps: float, time_interval: str = None, step: int = None,
           fmt: str = 'gif', path: str = None, n_tail: int = 0):
    """Export monitor screen with given time interval. Syntax:
     `anit [fps] [time_interval] [format] [path] [n_tail]`

    Examples
    --------
      anit 5 step=10
      anit 8 10000:12000 path=/home/william/gifs
      anit 10 fmt=mp4 n_tail=10

    Arguments
    ---------
    fps: Frame per second
    time_interval: Time interval (in seconds) to create animation.
                   E.g., 10000:20000.
    step: window sliding step. If not provided, will be set according to
          self.settable_attributes['step']
    fmt: Format of exported file, can be 'gif' or 'mp4'.
         If `fmt` is `mp4`, ffmpeg must be installed.
    path: Path to save the file
    n_tail: A workaround to avoid losing last few frames when export mp4
    """
    import re

    # -------------------------------------------------------------------------
    #  Create scripts
    # -------------------------------------------------------------------------
    ss = self._selected_signal
    ds = ss.dominate_signal

    # Check `time_interval` if provided
    if time_interval is not None:
      if re.match('^\d+:\d+$', time_interval) is None:
        raise ValueError('!! Illegal time interval `{}`'.format(time_interval))
      begin, end = [int(n) for n in time_interval.split(':')]
    else: begin, end = ds.ticks[0], ds.ticks[-1]

    # Get window parameters
    win_dur = ss.window_duration
    if step is None: step = win_dur * self.get('step')

    # Calculate script length and create scripts
    num = int((end - begin - win_dur) / step)
    begins = np.linspace(begin, end - win_dur, num=num)
    scripts = [lambda _t=t: self.goto(_t) for t in begins]
    # -------------------------------------------------------------------------
    #  Call animate to create animation
    # -------------------------------------------------------------------------
    self.pictor.animate(fps, scripts, fmt=fmt, path=path, n_tail=n_tail)

  # endregion: Public Methods

  # region: Commands and Shortcuts

  def move_window(self, direction=1, go_extreme=False):
    if self._selected_signal is None: return
    self._selected_signal.move_window(direction * self.get('step'), go_extreme)
    self.refresh()

  def goto_position(self, position: int):
    if self._selected_signal is None: return
    self._selected_signal.start_position = position
    self.refresh()

  def set_win_size(self, multiplier: int):
    if self._selected_signal is None: return
    self._selected_signal.set_window_size(multiplier)
    self.refresh()

  def set_win_duration(self, duration: float):
    """Set window duration (in secs)."""
    if self._selected_signal is None: return
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

  def add_channels(self, channels: str):
    channels = self.displayed_channels + [
      self.hidden_channels[int(i) - 1] for i in channels.split(',')]
    self.set('channels', ','.join(channels))
  ac = add_channels

  def highlight(self, id: int = 0):
    # Sanity check
    assert isinstance(id, int)

    # Get number of all displayed channels
    channel_str = self.get('channels')
    if channel_str == '*': N = len(self.channel_list)
    else: N = len(channel_str.split(','))

    # Set id
    self.set('hl', id % (N + 1))
  hl = highlight

  def remove_highlighted_channel(self):
    id = self.get('hl')
    if id == 0: return
    channels = self.get('channels')

    if channels == '*': channels = self.channel_list
    else: channels = channels.split(',')

    # Remove channel
    assert isinstance(channels, list)
    channels.pop(id - 1)
    self.set('channels', ','.join(channels))

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
    self.register_a_shortcut('s', lambda: self.flip('smart_scale'),
                             description='Toggle smart scale')
    self.register_a_shortcut(
      'bracketleft', lambda: self.set('xi', 0.5 * self.get('xi')), 'Halve xi')
    self.register_a_shortcut(
      'bracketright', lambda: self.set('xi', 2 * self.get('xi')), 'Doulbe xi')

    self.register_a_shortcut(
      'J', lambda: self.hl(self.get('hl') + 1), 'Highlight next channel')
    self.register_a_shortcut(
      'K', lambda: self.hl(self.get('hl') - 1), 'Highlight previous channel')
    self.register_a_shortcut('Return', self.hl, 'Cancel highlighting')

    self.register_a_shortcut('x', self.remove_highlighted_channel,
                             'Remove highlighted channel')

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
