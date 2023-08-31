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
from roma import console
from roma import Nomear

import numpy as np



class DigitalSignal(Nomear):
  """Digital signal(s) organized in a channel-last format
  """

  def __init__(self, data: np.ndarray, sfreq=None, ticks=None,
               channel_names=None, label='Blank Label', off_set=0., **kwargs):
    """Multi-channel digital signal stored in a 'channel-last' format.
    At least one of sampling_frequency and ticks should be provided while
    instantiating.

    Parameters:
    :param data: np.ndarray, will be reshaped to [L, C] if only one
           sequence with 1-D shape is provided
    :param sfreq: a float number
    :param channel_names - if provided, len(channel_names) should be C
    :param ticks: if provided, len(ticks) should be equal to len(sequence)
    :param off_set: used for calculating ticks if ticks are not provided
    :param label: name of this set of digital signal
    """
    # Check sequence, make sure its shape is [<seq_len>, <num_channels>]
    self.data = self._check_sequence(data, **kwargs)

    # Set sampling frequency and ticks
    self.sfreq, self._ticks = self._check_sfreq_and_ticks(sfreq, ticks)

    # Set channel_names
    self.channels_names = self._check_channel_names(channel_names)

    # Set other attributes
    self.off_set = off_set
    self.label = label
    self.kwargs = kwargs

  # region: Properties

  @Nomear.property()
  def ticks(self):
    if isinstance(self._ticks, np.ndarray): return self._ticks
    assert isinstance(self.sfreq, (int, float))
    return np.arange(len(self.data)) / self.sfreq + self.off_set

  @property
  def length(self): return len(self.data)

  @property
  def num_channels(self): return self.data.shape[1]

  @Nomear.property()
  def name_tick_data_list(self):
    """Channels should not be added after this property has been called"""
    return [(name, self.ticks, self.data[:, i])
            for i, name in enumerate(self.channels_names)]

  # endregion: Properties

  # region: Special Methods

  def __str__(self):
    return (f'DigitalSignal(label=`{self.label}`, ' +
            f'sfreq={self.sfreq}, data.shape={self.data.shape})')

  def __getitem__(self, item):
    # Case 1, e.g., ds[10:20]
    if isinstance(item, slice):
      start_time = self.ticks[0] if item.start is None else item.start
      end_time = self.ticks[-1] if item.stop is None else item.stop
      start_index = np.argwhere(self.ticks >= start_time).ravel()[0]
      stop_index = np.argwhere(self.ticks < end_time).ravel()[-1]

      if self._ticks is not None:
        _ticks = self._ticks[start_index:stop_index+1]
        off_set = 0.
      else:
        _ticks = None
        off_set = self.ticks[start_index]
      return DigitalSignal(
        self.data[start_index:stop_index+1], self.sfreq, _ticks,
        self.channels_names, self.label, off_set=off_set)

    # Case 2, e.g., ds['EEG']
    if item not in self.channels_names:
      raise KeyError(f'!! Channel `{item}` not found')
    return self.data[:, self.channels_names.index(item)]

  # endregion: Special Methods

  # region: Public Methods

  def add_channel(self, sequence: np.ndarray, name=None):
    assert not self.in_pocket('name_tick_data_list')
    if len(sequence.shape) == 1:
      sequence = sequence.reshape(shape=[-1, 1])
    assert len(sequence.shape) == 2 and len(sequence) == self.length
    # Add sequence
    self.data = np.concatenate([self.data, sequence], axis=-1)

    # Add name
    if name is None: name = f'Channel-{self.length + 1}'
    self.channels_names.append(name)

  def get_channel_percentile(self, name, percentile):
    assert 0 <= percentile <= 100
    key = (name, percentile)
    if not self.in_pocket(key):
      self.put_into_pocket(key, np.percentile(self[name], percentile),
                           local=True)
    return self.get_from_pocket(key)

  @staticmethod
  def preprocess_iqr(x: np.ndarray, iqr=1, median=0, max_abs_deviation=20,
                     labels=None):
    """Rescale data (shape=[L, C]) TODO: tooooooo slow"""
    assert median == 0

    # Subtract median from each element
    x = x - np.median(x, axis=0)
    current_iqr = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)

    # Handle 0 iqr issue
    zero_mask = current_iqr == 0
    if any(zero_mask):
      C = x.shape[1]
      if labels is None: labels = [f'Channel-{i+1}' for i in range(C)]
      assert len(labels) == C
      current_iqr[zero_mask] = 1.
      mask_labels = [l for m, l in zip(zero_mask, labels) if m]
      console.warning(
        f'Failed to rescale {",".join(mask_labels)} since IQR is 0.')

    x = x / current_iqr * iqr

    # Clip outliers if necessary
    if max_abs_deviation is not None and max_abs_deviation > 0:
      M = max_abs_deviation * iqr
      x = np.clip(x, -M, M)

    return x

  # endregion: Public Methods

  # region: Private Methods

  @staticmethod
  def _check_sequence(data, **kwargs):
    assert isinstance(data, np.ndarray)
    if len(data.shape) == 1: data = data.reshape([-1, 1])
    assert len(data.shape) == 2
    # Make sure sequence is in a channel-last format
    if kwargs.get('length_over_channels', True):
      assert data.shape[0] > data.shape[1]
    return data

  def _check_channel_names(self, channel_names):
    if channel_names is None:
      return [f'Channel-{i+1}' for i in range(self.data.shape[1])]
    if isinstance(channel_names, str): channel_names = [channel_names]
    assert len(channel_names) == self.data.shape[1]
    return channel_names

  def _check_sfreq_and_ticks(self, sfreq, ticks):
    if ticks is not None:
      ticks = np.array(ticks).ravel()
      if len(ticks) != self.length: raise AssertionError(
        '!! data length does not match `ticks` length')
      tick_freq = (len(ticks) - 1) / (ticks[-1] - ticks[0])
      if sfreq is None: sfreq = tick_freq
      elif abs(sfreq - tick_freq) > 1e-6: raise AssertionError(
        '!! `sfreq` does not match provided `ticks`')
    elif sfreq is None: raise ValueError(
      '!! At least one of `sfreq` or `ticks` should be provided')

    return sfreq, ticks

  # endregion: Private Methods

  # region: Operations

  # region: STFT

  @staticmethod
  def stft():
    """Performs Short-Time Fourier Transformation on signal.

    Reference: https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/    :return:
    """
    pass

  # endregion: STFT

  # endregion: Operations



if __name__ == '__main__':
  ds = DigitalSignal(np.random.random(size=(50, 3)), sfreq=1.0, off_set=1000,
                     channel_names=('A', 'B', 'C'))
  print(ds.ticks[0])
  print(ds.ticks.shape)
  print(ds['B'].shape)

  print(ds[1010:1020].length)



