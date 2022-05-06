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

  # endregion: Properties

  # region: Private Methods

  def _register_default_key_events(self):
    # Four directions
    self.shortcuts.register_key_event(
      ['j', 'down'],
      lambda: self.set_cursor(self.Keys.OBJECTS, 1, refresh=True),
      description='Next object', color='yellow')
    self.shortcuts.register_key_event(
      ['k', 'up'],
      lambda: self.set_cursor(self.Keys.OBJECTS, -1, refresh=True),
      description='Previous object', color='yellow')
    self.shortcuts.register_key_event(
      ['h', 'left'],
      lambda: self.set_cursor(self.Keys.PLOTTERS, 1, refresh=True),
      description='Next plotter', color='yellow')
    self.shortcuts.register_key_event(
      ['l', 'right'],
      lambda: self.set_cursor(self.Keys.PLOTTERS, -1, refresh=True),
      description='Previous plotter', color='yellow')

  def _set_default_layout(self):
    # Pack all widgets to self
    self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    # Finally, pack self to root
    self.pack()

  # endregion: Private Methods

  # region: Public Methods

  def add_plotter(self, plotter: Callable, index: int = -1):
    if not callable(plotter):
      raise ValueError('!! A plotter should be callable')
    if not isinstance(plotter, Plotter): plotter = Plotter(plotter)
    self.add_to_axis(self.Keys.PLOTTERS, plotter, index=index)

  def refresh(self):
    # Refresh title
    self.title = f'{self.cursor_string} {self.static_title}'
    # Refresh canvas
    self.canvas.refresh()

  # endregion: Public Methods

  # region: Builtin Commands

  def yell(self, text):
    """Doc string ..."""
    self.title = text

  # endregion: Builtin Commands
