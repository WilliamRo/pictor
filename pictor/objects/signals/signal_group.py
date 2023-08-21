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
from collections import OrderedDict
from roma import Nomear
from typing import List
from pictor.objects.signals.digital_signal import DigitalSignal

import numpy as np



class SignalGroup(Nomear):
  """A SignalGroup maintains a list of digital signals.
  """

  def __init__(self, signals, label=None, **properties):
    # Wrap data up if necessary
    if isinstance(signals, np.ndarray): signals = DigitalSignal(signals)
    if isinstance(signals, DigitalSignal): signals = [signals]

    self.digital_signals: List[DigitalSignal] = signals
    self.label = label
    self.properties = properties
    self.annotations = OrderedDict()

  # region: Properties

  @Nomear.property()
  def name_tick_data_list(self):
    res = []
    for ds in self.digital_signals:
      res.extend(ds.name_tick_data_list)
    return res

  @Nomear.property()
  def name_tick_data_dict(self):
    res = {}
    for ds in self.digital_signals:
      for name, tick, data in ds.name_tick_data_list: res[name] = (tick, data)
    return res

  @Nomear.property()
  def channel_signal_dict(self):
    """Returns {name: ds} dict in which ds contains `name` channel"""
    res = {}
    for ds in self.digital_signals:
      for name in ds.channels_names: res[name] = ds
    return res

  @property
  def signal_labels(self): return [ds.label for ds in self.digital_signals]

  @property
  def dominate_signal(self) -> DigitalSignal:
    return list(sorted(self.digital_signals, key=lambda ds: ds.length))[-1]

  @property
  def max_length(self): return self.dominate_signal.length

  @property
  def total_duration(self):
    ticks = self.dominate_signal.ticks
    return ticks[-1] - ticks[0]

  # endregion: Properties

  # region: Public Methods

  # region: Special Methods

  def truncate(self, start_time=0, end_time=-1, return_new_sg=False):
    """Truncate signal from `start_time` to `end_time`"""
    # Truncate digital signals
    digital_signals = [
      ds[start_time:end_time] for ds in self.digital_signals]
    # Truncate annotations
    annotations = {k: a.truncate(start_time, end_time)
                   for k, a in self.annotations.items()}

    if return_new_sg:
      sg = SignalGroup(digital_signals, self.label, **self.properties)
      sg.annotations = annotations
      return sg
    else:
      self.digital_signals = digital_signals
      self.annotations = annotations

  def __getitem__(self, item):
    if item not in self.channel_signal_dict:
      raise KeyError(f'!! Signal label `{item}` not found')
    ds = self.channel_signal_dict[item]
    return ds[item]

  # endregion: Special Methods

  def get_channel_percentile(self, name, percentile):
    ds = self.channel_signal_dict[name]
    return ds.get_channel_percentile(name, percentile)

  def set_annotation(self, key, intervals, annotations, labels):
    # Set annotation to dictionary
    self.annotations[key] = Annotation(intervals, annotations, labels)

  # endregion: Public Methods

  # region: Static Methods

  # endregion: Static Methods



class Annotation(Nomear):

  def __init__(self, intervals, annotations=None, labels=None):
    """Construct an Annotation
    :param intervals: float or a list of tuples of (start_time, end_time)
    :param annotations: if provided, should be a list or 1D numpy array of
                        integers
    :param labels: if provided, should be a list of strings,
                   e.g., ['W', 'REM', 'N1', 'N2', 'N3'] or a string
    """
    # Sanity check
    if isinstance(intervals, (int, float)):
      assert intervals > 0 and annotations is not None
      intervals = [(i * intervals, (i + 1) * intervals)
                   for i, _ in enumerate(annotations)]

    if annotations is not None:
      assert len(intervals) == len(annotations)
    # assert max(annotations) <= len(labels) - 1

    self.intervals = intervals
    self.annotations = annotations
    self.labels = labels

  @property
  def is_for_events(self): return self.annotations is None

  @Nomear.property()
  def labels_seen(self):
    if self.is_for_events: return None
    return self.labels[:max(self.annotations) + 1]

  @Nomear.property()
  def curve(self):
    ticks, values = [], []
    for inter, anno in zip(self.intervals, self.annotations):
      if ticks and values[-1] == anno:
        ticks[-1] = inter[-1]
        continue
      ticks.extend(list(inter))
      values.extend([anno] * 2)
    return np.array(ticks), np.array(values)

  def truncate(self, start_time, end_time):
    assert start_time < end_time

    inter, anno = [], (None if self.is_for_events else [])
    for i, interval in enumerate(self.intervals):
      if interval[0] >= end_time or interval[1] <= start_time: continue
      inter.append((max(interval[0], start_time), min(interval[1], end_time)))
      if not self.is_for_events: anno.append(self.annotations[i])

    return Annotation(inter, anno, labels=self.labels)

  def get_ticks_values_for_plot(self, start_time, end_time):
    """This method is for stage annotation only"""
    # Sanity check
    assert start_time < end_time
    ticks, values = self.curve

    # Locate start and end indices using np.argmax
    start_i = np.argmax(ticks >= start_time)
    start_i = max(start_i - 1, 0)
    end_i = np.argmax(ticks >= end_time)

    # Handle edge cases
    if end_i <= start_i: end_i = len(ticks)

    return ticks[start_i:end_i+1], values[start_i:end_i+1]
