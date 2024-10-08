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
from fnmatch import fnmatch
from pictor.objects.signals.digital_signal import DigitalSignal
from pictor.objects.signals.signal_group import SignalGroup



class EEG(SignalGroup):

  # region: Properties

  @property
  def digital_signal(self) -> DigitalSignal:
    assert len(self.digital_signals) == 1
    return self.digital_signals[0]

  # endregion: Properties

  # region: Public Methods#

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

    raise KeyError(f'!! Signal label `{item}` not found')

  # endregion: Public Methods
