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

from roma import Easel
from typing import Callable

import tkinter as tk



class Pictor(Easel):

  class Keys:
    OBJECTS = 'ObJeCtS'
    PLOTTERS = 'PlOtTeRs'

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

  # region: Properties

  @property
  def objects(self): return self.axes[self.Keys.OBJECTS]

  @objects.setter
  def objects(self, value):
    self.set_to_axis(self.Keys.OBJECTS, value, overwrite=True)

  @property
  def active_plotter(self) -> Plotter:
    plotter = self.get_element(self.Keys.PLOTTERS)
    if plotter is None: return self.canvas.default_plotter
    return plotter

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
    self.pack()

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

  def refresh(self):
    # Refresh title
    self.title = f'{self.cursor_string} {self.static_title}'
    # Refresh canvas
    self.canvas.refresh()

  # endregion: Public Methods

  # region: Presets

  @staticmethod
  def image_viewer(title='Image Viewer', figure_size=(5, 5)):
    from .plotters import Retina
    p = Pictor(title=title, figure_size=figure_size)
    p.add_plotter(Retina())
    return p

  @staticmethod
  def signal_viewer(title='Signal Viewer', figure_size=(9, 3), outline_bar=True,
                    **kwargs):
    from .plotters import Oscilloscope
    from .widgets import OutlineBar

    p = Pictor(title=title, figure_size=figure_size)

    # Add an oscilloscope
    osc = Oscilloscope(**kwargs)
    p.add_plotter(osc)

    if outline_bar:
      # Add an OutlineBar
      bar = OutlineBar(p)
      bar.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
      # Link oscilloscope to outline bar
      osc.link_to_outline_bar(bar)

    return p

  # endregion: Presets
