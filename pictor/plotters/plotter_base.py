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
from collections import OrderedDict
from roma import console
from roma import Nomear
from typing import Callable, Optional

import inspect



class Plotter(Nomear):

  def __init__(self, func: Callable, pictor=None):
    from ..pictor import Pictor
    self.func: Callable = func
    self.pictor: Optional[Pictor] = None

    # Shortcut will be accessed by roma.art.Shortcut
    self.shortcuts = OrderedDict()
    self.register_shortcuts()

    # name: (value, data_type, description)
    self.settable_attributes = OrderedDict()

    # Register if provided
    if pictor is not None: self.register_to_master(pictor)

  # region: Properties

  @property
  def argument_keys(self):
    return list(inspect.signature(self.func).parameters.keys())

  @property
  def class_name(self): return self.__class__.__name__

  # endregion: Properties

  # region: Methods to be overwritten

  def register_shortcuts(self): pass

  # endregion: Methods to be overwritten

  # region: Private Methods

  def _check_attribute(self, key):
    if key not in self.settable_attributes: raise KeyError(
      '!! `{}` is not a settable attribute of {}'.format(key, self.class_name))

  # endregion: Private Methods

  # region: Public Methods

  def refresh(self): self.pictor.refresh()

  def register_to_master(self, pictor):
    """Register self to pictor"""
    self.pictor = pictor

  def register_a_shortcut(self, key: str, func: Callable, description,
                          color='yellow'):
    self.shortcuts[key] = (func, description, color)

  def new_settable_attr(self, name, default_value, _type, description):
    self.settable_attributes[name] = [default_value, _type, description]

  # endregion: Public Methods

  # region: Commands

  def set(self, key, value=None):
    if key == '??':
      # TODO: show a table of all settable attributes
      console.show_info(f'All settable attributes of `{self.class_name}`:')
      for k, v in self.settable_attributes.items():
        console.supplement(f'{k}({v[0]}, {v[1].__name__}): {v[2]}', level=2)
      return

    self._check_attribute(key)
    _type = self.settable_attributes[key][1]
    # Convert value accordingly
    if value in (None, 'none', 'None'): value = None
    else: value = _type(value)
    self.settable_attributes[key][0] = value

    console.show_status(f'{self.class_name}.{key} set to {value}')

    # Refresh canvas
    self.refresh()

  def get(self, key):
    self._check_attribute(key)
    return self.settable_attributes[key][0]

  def flip(self, key):
    """Flip a boolean value"""
    self._check_attribute(key)
    value, _type, _ = self.settable_attributes[key]
    if _type is not bool: raise TypeError(
      '!! {}.{} should be boolean'.format(self.class_name, key))

    self.set(key, not value)

  # endregion: Commands

  # region: Special Functions

  def __call__(self):
    # Try to get key-word arguments for method according to its signature
    kwargs = {}
    for k in self.argument_keys:
      if k in ('obj', 'x'):
        kwargs[k] = self.pictor.get_element(self.pictor.Keys.OBJECTS)
      elif k in ('i', 'index'):
        kwargs[k] = self.pictor.cursors[self.pictor.Keys.OBJECTS]
      elif k in ('canvas', ):
        kwargs[k] = self.pictor.canvas
      elif k in ('fig', 'figure'):
        kwargs[k] = self.pictor.canvas.figure
      elif k in ('ax', 'axes', 'ax2d', 'axes2d'):
        kwargs[k] = self.pictor.canvas.axes2D
      elif k in ('ax3d', 'axes3d'):
        kwargs[k] = self.pictor.canvas.axes3D
      elif hasattr(self, k):
        kwargs[k] = getattr(self, k)

    # Call function
    self.func(**kwargs)

  # endregion: Special Functions
