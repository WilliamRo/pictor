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
from roma import Nomear
from typing import Optional, Union

import numpy as np
import random



class DigitalSignal(Nomear):
  """Digital signal(s) organized in a channel-last format
  """

  def __init__(self, sequence: np.ndarray, ticks=None, channel_names=None,
               label='Blank Label', **kwargs):
    """Parameters:
    sequence: np.ndarray, will be reshaped to [L, C] if only one sequence with
              1-D shape is provided
    channel_names: if provided, len(channel_names) should be C
    ticks: if provided, len(ticks) should be equal to len(sequence)
    label: name of this set of digital signal
    """
    # Check sequence, reshape if necessary
    assert isinstance(sequence, np.ndarray)
    if len(sequence.shape) == 1: sequence = sequence.reshape([-1, 1])
    assert len(sequence.shape) == 2
    # Make sure sequence is in a channel-last format
    if kwargs.get('length_over_channels', True):
      assert sequence.shape[0] > sequence.shape[1]
    # sequences.shape = [<seq_len>, <num_channels>]
    self.sequence = sequence

    # Set ticks
    if ticks is None: ticks = range(self.length)
    ticks = np.array(ticks).ravel()
    assert len(ticks) == self.length
    self.ticks = ticks

    # Set channel_names
    if channel_names is None:
      channel_names = [f'Channel-{i+1}' for i in range(self.num_channels)]
    if isinstance(channel_names, str): channel_names = [channel_names]
    assert len(channel_names) == self.num_channels
    self.channels_names = list(channel_names)

    self.label = label

  # region: Properties

  @property
  def length(self): return len(self.sequence)

  @property
  def num_channels(self): return self.sequence.shape[1]

  @Nomear.property()
  def name_tick_data_list(self):
    """Channels should not be added after this property has been called"""
    return [(name, self.ticks, self.sequence[:, i])
            for i, name in enumerate(self.channels_names)]

  # endregion: Properties

  # region: Special Methods

  def __getitem__(self, item):
    if item not in self.channels_names:
      raise KeyError(f'!! Channel `{item}` not found')
    return self.sequence[:, self.channels_names.index(item)]

  # endregion: Special Methods

  # region: Public Methods

  def add_channel(self, sequence: np.ndarray, name=None):
    assert not self.in_pocket('name_tick_data_list')
    if len(sequence.shape) == 1:
      sequence = sequence.reshape(shape=[-1, 1])
    assert len(sequence.shape) == 2 and len(sequence) == self.length
    # Add sequence
    self.sequence = np.concatenate([self.sequence, sequence], axis=-1)

    # Add name
    if name is None: name = f'Channel-{self.length + 1}'
    self.channels_names.append(name)

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



if __name__ == '__main__':
  ds = DigitalSignal(np.random.random(size=(50, 3)),
                     channel_names=('A', 'B', 'C'))
  print(ds['B'].shape)



