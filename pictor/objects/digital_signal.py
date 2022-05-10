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
# ===-=====================================================================-====
from typing import Optional, Union

import numpy as np
import random



class DigitalSignal(object):
  """
  data y = |.................................................|, length = L
           |-- window_size --|

  """

  def __init__(self, sequence: np.ndarray,
               indices: Optional[np.ndarray] = None):
    self.y = sequence
    if indices is None: indices = np.arange(len(sequence))
    self.x = indices

    # Display options
    self.window_size = self.length
    self.starting_index = 0

  # region: Properties

  @property
  def length(self): return len(self.y)

  @property
  def xlim(self):
    return (max(self.starting_index, 0),
            min(self.starting_index + self.window_size - 1, self.length - 1))

  # endregion: Properties

  # region: Public Methods

  def move_window(self, step_ratio):
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

  # endregion: Public Methods

  # region: Static Methods

  @staticmethod
  def sinusoidal(x: np.ndarray, omega: Union[float, list] = 1.0,
                 phi: float = 0.0, noise_db=None, max_truncate_ratio=0.0):
    assert noise_db is None

    if isinstance(omega, (float, int)):
      # Truncate if permitted
      if 0 < max_truncate_ratio < 1:
        min_index = int((1 - max_truncate_ratio) * len(x))
        x = x[:random.randint(min_index, len(x) + 1)]
      # TODO: add noise
      return np.sin(omega * x + phi)

    return np.concatenate([DigitalSignal.sinusoidal(
      x, om, phi, noise_db, max_truncate_ratio) for om in omega])

  # endregion: Static Methods
