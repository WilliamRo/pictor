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
# ===-=========================================================================-
from pictor.objects.signals import DigitalSignal, SignalGroup
from typing import Optional, Union

import numpy as np
import random



class Scrolling(SignalGroup):
  """
  data y = |.................................................|, length = L
           |-- window_size --|
  """

  class Keys:
    window_size = 'Scrolling:window_size'
    start_position = 'Scrolling:start_position'

  # region: Properties

  @property
  def window_size(self):
    return self.get_from_pocket(self.Keys.window_size, default=None)

  @window_size.setter
  def window_size(self, value):
    assert isinstance(value, int) and value > 0
    self.put_into_pocket(self.Keys.window_size,
                         min(value, self.max_length), exclusive=False)

  @property
  def start_position(self):
    return self.get_from_pocket(self.Keys.start_position,
                                initializer=lambda: 0.0)

  @start_position.setter
  def start_position(self, value):
    assert 0 <= value <= 1.0
    self.put_into_pocket(self.Keys.start_position, value, exclusive=False)

  # @property
  # def xlim(self): return (self.x[self.starting_index],
  #                         self.x[self.starting_index + self.window_size - 1])

  # @property
  # def window_location_pct(self):
  #   p_min = self.starting_index / self.length
  #   p_max = p_min + self.window_size / self.length
  #   return p_min, p_max

  # endregion: Properties

  # region: Public Methods

  def get_channels(self, channels: str):
    """channels can be
       (1) *: for all channels
       (2) names split by comma, e.g., `EEG,ECG`
       (3) list of channels
    """
    return [(name, x, y) for name, x, y in self.name_tick_data_list
            if channels == '*' or name in channels]

  def move_window(self, step_ratio, go_extreme=False):
    # If go extreme, go home if step_ratio < 0, otherwise go end
    if go_extreme: self.starting_index = (
        0 if step_ratio < 0 else self.length - self.window_size)

    # Calculate step
    step = int(step_ratio * self.window_size)
    # Move window
    self.starting_index = self.starting_index + step
    # Handle left edge condition
    self.starting_index = max(self.starting_index, 0)
    # Handle right edge condition
    self.starting_index = min(
      self.starting_index, self.length - self.window_size)

  def set_window_size(self, multiplier):
    ws = int(self.window_size * multiplier)
    ws = max(min(ws, self.length), 10)
    self.window_size = ws
    # Call move_window(0) to prevent window out of bound
    self.move_window(0)

  # endregion: Public Methods

  # region: Static Methods

  # endregion: Static Methods
