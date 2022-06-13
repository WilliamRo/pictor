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
from roma import Nomear
from typing import List
from pictor.objects.signals.digital_signal import DigitalSignal

import numpy as np



class SignalGroup(Nomear):
  """A SignalGroup maintains a list of digital signals.
  """

  def __init__(self, signals, label=None):
    # Wrap data up if necessary
    if isinstance(signals, np.ndarray): signals = DigitalSignal(signals)
    if isinstance(signals, DigitalSignal): signals = [signals]

    self.digital_signals: List[DigitalSignal] = signals
    self.label = label

  # region: Properties

  @Nomear.property()
  def name_tick_data_list(self):
    res = []
    for ds in self.digital_signals:
      res.extend(ds.name_tick_data_list)
    return res

  @property
  def signal_labels(self): return [ds.label for ds in self.digital_signals]

  @property
  def dominate_signal(self) -> DigitalSignal:
    return list(sorted(self.digital_signals, key=lambda ds: ds.length))[-1]

  @property
  def max_length(self): return self.dominate_signal.length
  # def max_length(self): return max([ds.length for ds in self.digital_signals])

  # endregion: Properties

  # region: Public Methods

  # region: Special Methods

  def __getitem__(self, item):
    if item not in self.signal_labels:
      raise KeyError(f'!! Signal label `{item}` not found')
    return self.digital_signals[self.signal_labels.index(item)]

  # endregion: Special Methods

  # endregion: Public Methods

  # region: Static Methods

  # endregion: Static Methods
