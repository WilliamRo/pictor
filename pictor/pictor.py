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
from .plotters import Plotter
from .widgets import Canvas
from .plugins.database import Database
from .plugins.studio import Studio

from roma import Easel
from roma import Nomear
from typing import Callable

import tkinter as tk



class Pictor(Easel, Database, Studio):

  class Keys:
    OBJECTS = 'ObJeCtS'
    PLOTTERS = 'PlOtTeRs'
    LABELS = 'LaBeLs'
    TITLE_SUFFIX = 'TiTlE_sUfFiX'
    ALLOW_MAIN_THREAD_REFRESH = 'MaIn_ThReAd_ReFrEsH'

  def __init__(self, title='Pictor', figure_size=(5, 5)):
    # Call parent's constructor
    super(Pictor, self).__init__()

    # Set window title
    self.static_title = title

    # Register widgets
    self.canvas = Canvas(self, figure_size=figure_size)

    # Create dimensions for objects and layers
    self.create_dimension(self.Keys.OBJECTS)
    self.create_dimension(self.Keys.PLOTTERS)

    # Register common events
    self._register_default_key_events()

    # Set layout
    self._set_default_layout()

    # TODO: beta
    self.on_command_text_changed = self.default_on_command_text_changed
    self.busy_plotter = None

  # region: Properties

  @property
  def objects(self): return self.axes[self.Keys.OBJECTS]

  @objects.setter
  def objects(self, value):
    self.set_to_axis(self.Keys.OBJECTS, value, overwrite=True)

  @property
  def labels(self):
    return self.get_from_pocket(self.Keys.LABELS, default=None)

  @labels.setter
  def labels(self, values):
    assert len(values) == len(self.objects)
    self.put_into_pocket(self.Keys.LABELS, values)

  @property
  def plotters(self): return self.axes[self.Keys.PLOTTERS]

  @property
  def active_plotter(self) -> Plotter:
    if self.busy_plotter is not None: return self.busy_plotter
    plotter = self.get_element(self.Keys.PLOTTERS)
    if plotter is None: return self.canvas.default_plotter
    return plotter

  @property
  def title_suffix(self) -> str:
    return self.get_from_pocket(self.Keys.TITLE_SUFFIX, '', put_back=False)

  @title_suffix.setter
  def title_suffix(self, suffix):
    self.put_into_pocket(self.Keys.TITLE_SUFFIX, suffix, exclusive=False)

  @property
  def command_hints(self) -> dict:
    ch = self._command_hints
    ch.update(self.active_plotter.command_hints)
    return ch

  @Nomear.property()
  def _command_hints(self) -> dict: return {}

  # endregion: Properties

  # region: Private Methods

  def _register_default_key_events(self):
    # Allow plotter shortcuts
    self.shortcuts.external_fetcher = self._get_plotter_shortcuts

    # Four directions
    self.shortcuts.register_key_event(
      ['j', 'Down'],
      lambda: self.set_cursor(self.Keys.OBJECTS, 1, refresh=True),
      description='Next object', color='yellow')
    self.shortcuts.register_key_event(
      ['k', 'Up'],
      lambda: self.set_cursor(self.Keys.OBJECTS, -1, refresh=True),
      description='Previous object', color='yellow')
    self.shortcuts.register_key_event(
      ['n'],
      lambda: self.set_cursor(self.Keys.PLOTTERS, 1, refresh=True),
      description='Next plotter', color='yellow')
    self.shortcuts.register_key_event(
      ['p'],
      lambda: self.set_cursor(self.Keys.PLOTTERS, -1, refresh=True),
      description='Previous plotter', color='yellow')

  def _set_default_layout(self):
    # Set main frame's background
    self.configure(background='white')
    # Pack all widgets to self
    self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    # Finally, pack self to root
    self.pack(fill=tk.BOTH, expand=True)

  def _get_attribute(self, key, default_value):
    """This method helps commander to access to plotters"""
    if hasattr(self, key): return getattr(self, key)
    return getattr(self.active_plotter, key, default_value)

  def _get_plotter_shortcuts(self):
    """This method help Shortcut to access to plotters"""
    return self.active_plotter.shortcuts

  # endregion: Private Methods

  # region: Public Methods

  def add_plotter(self, plotter: Callable, index: int = -1):
    if not callable(plotter):
      raise ValueError('!! A plotter should be callable')
    if not isinstance(plotter, Plotter):
      plotter = Plotter(plotter)

    # Set master
    plotter.register_to_master(self)
    self.add_to_axis(self.Keys.PLOTTERS, plotter, index=index)
    return plotter

  def config_plotter(self, index=0, **kwargs):
    plotter: Plotter = self.axes[self.Keys.PLOTTERS][index]
    for k, v in kwargs.items(): plotter.set(k, v, auto_refresh=False)

  def refresh(self, wait_for_idle=False):
    if not wait_for_idle and not self.get_from_pocket(
        self.Keys.ALLOW_MAIN_THREAD_REFRESH, default=True):
      return
    # Refresh title
    self.title = f'{self.cursor_string} {self.static_title}{self.title_suffix}'
    # Refresh canvas
    self.canvas.refresh(wait_for_idle)

  def allow_main_thread_refreshing(self, val: bool):
    self.put_into_pocket(self.Keys.ALLOW_MAIN_THREAD_REFRESH,
                         val, exclusive=False)

  def busy(self, message: str, auto_refresh=True):
    return BusyIndicator(self, message, auto_refresh)

  # endregion: Public Methods

  # region: Commands

  def set_cursor(self, key: str, step: int = 0, cursor=None, refresh=False):
    super().set_cursor(key, step, cursor, refresh)

    # TODO: [patch] temporal workaround to fix hint issue
    if key == self.Keys.PLOTTERS:
      self.active_plotter.register_to_master(self)

  def set_object_cursor(self, i: int):
    self.set_cursor(self.Keys.OBJECTS, cursor=i - 1, refresh=True)

  def set_plotter_cursor(self, i: int):
    self.set_cursor(self.Keys.PLOTTERS, cursor=i - 1, refresh=True)

  so = set_object_cursor
  sp = set_plotter_cursor

  # endregion: Commands

  # region: Presets

  @staticmethod
  def image_viewer(title='Image Viewer', figure_size=(5, 5)):
    from .plotters import Retina
    p = Pictor(title=title, figure_size=figure_size)
    p.add_plotter(Retina())
    return p

  @staticmethod
  def signal_viewer(title='Signal Viewer', figure_size=(9, 3), **kwargs):
    from .plotters import Oscilloscope

    p = Pictor(title=title, figure_size=figure_size)

    # Add an oscilloscope
    osc = Oscilloscope(**kwargs)
    p.add_plotter(osc)

    return p

  # endregion: Presets



class BusyIndicator(Nomear):

  def __init__(self, pictor: Pictor, message, auto_refresh=True):
    self.pictor: Pictor = pictor
    self.message = message
    self.auto_refresh = auto_refresh

  def show_busy_message(self, ax, fig):
    Plotter.show_text(self.message, ax, fig)

  def __enter__(self):
    self.pictor.busy_plotter = Plotter(self.show_busy_message,
                                       pictor=self.pictor)
    if self.auto_refresh: self.pictor.refresh()

  def __exit__(self, *args):
    self.pictor.busy_plotter = None
    if self.auto_refresh: self.pictor.refresh()