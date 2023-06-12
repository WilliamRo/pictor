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
from pictor.objects.signals import DigitalSignal, SignalGroup, Annotation
from typing import Optional, Union, List

import numpy as np



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
    """A value in [0, 1]"""
    return self.get_from_pocket(self.Keys.window_size,
                                initializer=lambda: 1.0)

  @window_size.setter
  def window_size(self, value):
    assert isinstance(value, float) and value > 0
    self.put_into_pocket(
      self.Keys.window_size, min(value, 1.0), exclusive=False)

  @property
  def window_duration(self):
    return self.total_duration * self.window_size

  @property
  def start_position(self):
    return self.get_from_pocket(self.Keys.start_position,
                                initializer=lambda: 0.0)

  @start_position.setter
  def start_position(self, value):
    """A value in [0, 1]"""
    # Handle edge condition
    value = min(max(value, 0), 1.0 - self.window_size)
    self.put_into_pocket(self.Keys.start_position, value, exclusive=False)

  # endregion: Properties

  # region: Public Methods

  def get_channels(self, channels: str, max_ticks=None):
    """channels can be
       (1) *: for all channels
       (2) names split by comma, e.g., `EEG,ECG`
       (3) list of channels
    """
    res = []
    if channels == '*': name_tick_data_list = self.name_tick_data_list
    else: name_tick_data_list = [
      (name, *self.name_tick_data_dict[name])
      for name in channels.split(',') if name in self.name_tick_data_dict]

    for name, x, y in name_tick_data_list:
      assert len(x) == len(y)
      start_i = int(len(x) * self.start_position)  # (1)
      end_i = start_i + int(self.window_size * len(x)) + 1

      # Apply max_ticks option if given
      if isinstance(max_ticks, int) and 0 < max_ticks < end_i - start_i:
        step = (end_i - start_i) // max_ticks
      else: step = 1
      res.append((name, x[start_i:end_i:step], y[start_i:end_i:step]))
    return res

  def get_annotation(self, key, start_time, end_time):
    if key not in self.annotations: return None
    anno: Annotation = self.annotations[key]

    if anno.is_for_events: return anno.truncate(start_time, end_time)

    ticks, values = anno.get_ticks_values_for_plot(start_time, end_time)
    # ticks, values = anno.curve
    return ticks, values, anno.labels_seen

  def move_window(self, step_ratio, go_extreme=False):
    # If go extreme, go home if step_ratio < 0, otherwise go end
    if go_extreme: self.start_position = (
      0.0 if step_ratio < 0 else 1.0 - self.window_size)

    # Calculate step
    step = step_ratio * self.window_size
    # Move window
    self.start_position = self.start_position + step

  def set_window_size(self, multiplier):
    ws = self.window_size * multiplier
    self.window_size = ws
    # Call move_window(0) to prevent window out of bound
    self.move_window(0)

  def set_window_duration(self, window_duration):
    if window_duration is None: return
    self.window_size = window_duration / self.total_duration

  # endregion: Public Methods
