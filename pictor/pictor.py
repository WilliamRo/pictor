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
from roma import Easel
from typing import Callable



class Pictor(Easel):

  class Keys:
    OBJECTS = 'ObJeCtS'
    PLOTTERS = 'PlOtTeRs'

  def __init__(self, title='Pictor'):
    # Call parent's constructor
    super(Pictor, self).__init__()

    # Set window title
    self.static_title = title

    # Create dimensions for objects and layers
    self.create_dimension(self.Keys.OBJECTS)
    self.create_dimension(self.Keys.PLOTTERS)

    # Register common events
    self._register_key_events()

  # region: Properties

  @property
  def objects(self): return self.axes[self.Keys.OBJECTS]

  @objects.setter
  def objects(self, value):
    self.set_to_axis(self.Keys.OBJECTS, value, overwrite=True)

  # endregion: Properties

  # region: Private Methods

  def _register_key_events(self):
    pass

  # endregion: Private Methods

  # region: Public Methods

  def add_plotter(self, plotter: Callable, index: int = -1):
    if not callable(plotter):
      raise ValueError('!! A plotter should be callable')
    self.add_to_axis(self.Keys.PLOTTERS, plotter, index=index)

  def refresh(self):
    self.title = f'{self.cursor_string} {self.static_title}'

  # endregion: Public Methods

  # region: Builtin Commands

  def yell(self, text):
    """Doc string ..."""
    self.title = text

  # endregion: Builtin Commands



if __name__ == '__main__':
  p = Pictor()
  p.objects = [1, 2, 3]
  p.show()

