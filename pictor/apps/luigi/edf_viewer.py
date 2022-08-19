from pictor import Pictor
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from pictor.plotters import Monitor
from roma import console
from typing import List

import numpy as np
import os



class EDFViewer(Pictor):

  def __init__(self, *file_paths, freq_modifier=None):
    self.file_paths = file_paths
    assert len(file_paths) > 0

    self._freq_modifier = freq_modifier

    # Call parent's constructor
    super(EDFViewer, self).__init__('Luigi', figure_size=(12, 8))

    # Add plotter
    self.monitor: Monitor = self.add_plotter(Monitor())

    # Set objects
    self._set_objects()

  # region: Private Methods

  def _set_objects(self):
    self.objects = [
      SignalGroup(self.read_edf_file(
        fp, freq_modifier=self._freq_modifier, verbose=True),
        label=f'{os.path.basename(fp)}') for fp in self.file_paths]

  # endregion: Private Methods

  # region: Public Methods

  @classmethod
  def read_edf_file(cls, fn: str,
                    channel_list: List[str] = None,
                    freq_modifier=None, verbose=False) -> List[DigitalSignal]:
    import pyedflib

    # Sanity check
    assert os.path.exists(fn)
    if verbose: console.show_status(f'Reading `{fn}`')

    signal_dict = {}
    with pyedflib.EdfReader(fn) as file:
      # Check channels
      all_channels = file.getSignalLabels()
      if channel_list is None: channel_list = all_channels
      # Read channels
      for i, channel_name in enumerate(channel_list):
        if verbose: console.print_progress(i, len(channel_list))
        # Get channel id
        chn = all_channels.index(channel_name)
        frequency = file.getSampleFrequency(chn)
        if callable(freq_modifier): frequency = freq_modifier(frequency)
        # Initialize an item in signal_dict if necessary
        if frequency not in signal_dict: signal_dict[frequency] = []
        # Read signal
        signal_dict[frequency].append((channel_name, file.readSignal(chn)))

    if verbose: console.show_status('Done')

    # Wrap data into DigitalSignals
    digital_signals = []
    for frequency, signal_list in signal_dict.items():
      ticks = np.arange(len(signal_list[0][1])) / frequency
      digital_signals.append(DigitalSignal(
        np.stack([x for _, x in signal_list], axis=-1), ticks=ticks,
        channel_names=[name for name, _ in signal_list],
        label=f'Freq=' f'{frequency}'))

    return digital_signals

  # endregion: Public Methods



if __name__ == '__main__':
  file_path = r'../../../../data/edfs/oliver-2022-08-17.edf'

  ev = EDFViewer(file_path, freq_modifier=lambda f: 1000 * f)
  ev.show()

