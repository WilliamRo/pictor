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
# =-===========================================================================-
from collections import OrderedDict
from fnmatch import fnmatch
from pictor.objects.signals.digital_signal import DigitalSignal
from pictor.objects.signals.signal_group import SignalGroup

import numpy as np



class EEG(SignalGroup):
  """A specialized SignalGroup for EEG signals. Note that only one digital
  signal is allowed in this class.
  """

  # region: Properties

  @property
  def sampling_frequency(self) -> float: return self.digital_signal.sfreq

  @property
  def digital_signal(self) -> DigitalSignal:
    assert len(self.digital_signals) == 1
    return self.digital_signals[0]

  # region: Components

  @SignalGroup.property(local=True)
  def components(self) -> dict:
    """Component dict for different frequency bands.

    Usage (given eeg: EEG):
      >> (s, (low, high), configs) = eeg.components['delta']
      >> s.shape    # (n_channels, n_samples)
      >> low, high  # (0.0, 4.0)
      >> configs    # configs = {'method': 'butter', 'N': 4, 'Wn': [0.0, 4.0],
                                 'output': 'ba', 'fs': 128.0, 'btype': 'band'}
    """
    return OrderedDict()

  # region: Built-in Components

  # Slow oscillations
  @property
  def SO(self): return self._ewr('SO', 0.5, 1)

  @property
  def delta(self): return self._ewr('delta', 0.1, 4)

  @property
  def delta_low(self): return self._ewr('delta_low', 0.1, 1.5)

  # K-complexes
  @property
  def delta_high(self): return self._ewr('delta_high', 1.5, 4)

  @property
  def theta(self): return self._ewr('theta', 4, 8)

  @property
  def alpha(self): return self._ewr('alpha', 8, 12)

  @property
  def alpha_1(self): return self._ewr('alpha_1', 8, 10)

  @property
  def alpha_2(self): return self._ewr('alpha_2', 10, 12)

  # Spindles
  @property
  def sigma(self): return self._ewr('sigma', 12, 16)

  @property
  def beta_1(self): return self._ewr('beta_1', 12, 20)

  @property
  def beta_2(self): return self._ewr('beta_2', 20, 30)

  @property
  def gamma_1(self): return self._ewr('gamma_1', 30, 45)

  # endregion: Built-in Components

  # endregion: Components

  # endregion: Properties

  # region: Public Methods#

  # region: Signal Decomposition

  def _extract_with_registration(
      self, key: str, low: float, high: float) -> np.ndarray:
    """Extract the frequency band signal with registration.
    Returns the signal of shape (n_channels, n_samples).
    """
    if key not in self.components:
      data, config = self.extract_band(low, high, return_config=True)
      self.components[key] = (data, (low, high), config)
    return self.components[key][0]
  _ewr = _extract_with_registration  # alias

  def extract_band(self, low: float, high: float, return_config=False):
    """Extract the frequency band signal from the EEG signal.
    Returns the signal of shape (n_channels, n_samples) with config if required.
    """
    signal_list = []
    for s in self.digital_signal.signals:
      component, config = self.extract_component(
        s, self.sampling_frequency, low, high, return_config=True)
      signal_list.append(component)
    extracted_data = np.stack(signal_list, axis=0)
    if return_config: return extracted_data, config
    return extracted_data

  def __getitem__(self, item):
    """Support syntax like EEG['delta'] to extract the delta band signals."""
    if hasattr(self, item): return getattr(self, item)
    return SignalGroup.__getitem__(self, item)

  # endregion: Signal Decomposition

  # region: MISC

  @classmethod
  def extract_eeg_channels_from_sg(
      cls, sg: SignalGroup, pattern: str = 'EEG *-*'):
    eeg = sg.extract_channels(list(sorted([
      chn for chn in sg.channel_names if fnmatch(chn, pattern)])))

    # The developer fully understands the implications and is certain it won't
    # cause unexpected behavior.
    eeg.__class__ = cls
    return eeg

  def re_reference(self, x, src: str, tgt: str):
    """Change the reference of the given EEG signal x.

    CLE = (Cz + A1 + A2) / 3
    LER = (A1 + A2) / 2

    (1) ??-CLE -> ??-LER given Cz-CLE
        Cz-CLE = (2*Cz - A1 - A2) / 3
        ??-CLE = ?? - (Cz + A1 + A2) / 3

        ??-LER = ?? - (A1 + A2) / 2
               = ??-CLE + (2*Cz - A1 - A2 ) / 6
               = ??-CLE + Cz-CLE / 2

    (2) ??-LER -> ??-CLE given Cz-LER
        Cz-LER = (2*Cz - A1 - A2) / 2
        ??-LER = ?? - (A1 + A2) / 2

        ??-CLE = ?? - (Cz + A1 + A2) / 3
               = ??-LER - (2*Cz - A1 - A2) / 6
               = ??-LER - Cz-LER / 3
    """
    if (src, tgt) == ('CLE', 'LER'):
      Cz_CLE_key = 'EEG Cz-CLE'
      assert Cz_CLE_key in self.channel_names
      return x + self[Cz_CLE_key] / 2
    elif (src, tgt) == ('LER', 'CLE'):
      Cz_LER_key = 'EEG Cz-LER'
      assert Cz_LER_key in self.channel_names
      return x - self[Cz_LER_key] / 3
    else: raise ValueError(f'!! Invalid reference pair: {src} -> {tgt}')

  def __getitem__(self, item: str):
    # (1) Return channel signal directly if exists
    if item in self.channel_signal_dict:
      ds = self.channel_signal_dict[item]
      return ds[item]
    elif fnmatch(item, '*-CLE'):
      x = self[item.replace('CLE', 'LER')]
      src_ref, tgt_ref = 'LER', 'CLE'
      return self.re_reference(x, src_ref, tgt_ref)
    elif fnmatch(item, '*-LER'):
      x = self[item.replace('LER', 'CLE')]
      src_ref, tgt_ref = 'CLE', 'LER'
      return self.re_reference(x, src_ref, tgt_ref)

    # (2) Return built-in component if exists
    if hasattr(self, item): return getattr(self, item)

    raise KeyError(f'!! Signal label `{item}` not found')

  # endregion: MISC

  # endregion: Public Methods
