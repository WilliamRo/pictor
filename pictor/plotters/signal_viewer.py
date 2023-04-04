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
import numpy as np
import matplotlib.pyplot as plt

from pictor.plotters.plotter_base import Plotter



class SignalViewer(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super(SignalViewer, self).__init__(self.plot_signal_and_spectrum, pictor)

    # Settable attributes
    self.new_settable_attr('slim', ',', str, 'Spectrum limit')
    self.new_settable_attr('max_freq', None, float, 'Maximum frequency')
    self.new_settable_attr('grid', True, bool, 'Option of showing grid')

  # region: Properties

  @property
  def spectrum_lim(self):
    ymin, ymax = self.get('slim').split(',')
    return [None if ymxn == '' else float(ymxn) for ymxn in (ymin, ymax)]

  # endregion: Properties

  # region: Plotting Method

  def _unpack(self, obj):
    # Allow obj to generate itself if it's callable
    if callable(obj): obj = obj(self)

    # Let sampling frequency to be 1 if not specified
    if isinstance(obj, np.ndarray):
      x, fs_or_ticks = obj, 1.0
    elif isinstance(obj, (list, tuple)):
      assert len(obj) == 2
      x, fs_or_ticks = obj

    # Generate ticks if not provided
    if isinstance(fs_or_ticks, np.ndarray):
      ticks = fs_or_ticks
      fs = (len(ticks) - 1) / (ticks[-1] - ticks[0])
    else:
      assert fs_or_ticks > 0
      fs = fs_or_ticks
      ticks = np.arange(len(x)) / fs

    return x, ticks, fs

  def plot_signal_and_spectrum(self, fig: plt.Figure, obj):
    # Get signal and its corresponding time ticks
    x, ticks, fs = self._unpack(obj)

    # (1) Plot time domain
    ax1: plt.Axes = fig.add_subplot(211)
    ax1.plot(ticks, x)

    ax1.set_xlim(ticks[0], ticks[-1])
    ax1.set_xlabel('Time')
    ax1.set_title(f'fs = {fs}')

    ax1.grid(self.get('grid'))

    # (2) Plot spectrum
    # Perform 1D-DFT
    freq = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / fs))
    X = np.fft.fftshift(np.fft.fft(x))

    ax2: plt.Axes = fig.add_subplot(212)
    abs_X = np.abs(X)
    ax2.plot(freq, 20 * np.log10(abs_X / np.max(abs_X)))

    # Set styles
    max_freq = self.get('max_freq')
    if max_freq is None: max_freq = freq[-1]
    ax2.set_xlim(0, max_freq)
    ax2.set_ylim(*self.spectrum_lim)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Decibels')

    ax2.grid(self.get('grid'))

  # endregion: Plotting Method



if __name__ == '__main__':
  p = SignalViewer.plot([], fig_size=(9, 6), show=False)
  sv: SignalViewer = p.plotters[0]
  sv.new_settable_attr('fs', 500.0, float, 'Sampling frequency')
  sv.new_settable_attr('T', 2.0, float, 'Sampling time')
  sv.new_settable_attr('phi', 0.0, float, 'Delay')
  sv.new_settable_attr('phi_step', 0.001, float, 'Step for changing delay')
  sv.new_settable_attr('omega', 5.0, float, 'Angular frequency')

  sv.set('slim', '-40, 60', False, verbose=False)
  sv.set('max_freq', 30, False, verbose=False)

  def change_phi(d=1.0):
    phi, step = sv.get('phi'), sv.get('phi_step')
    sv.set('phi', phi + d * step, verbose=False)

  sv.register_a_shortcut('n', lambda: change_phi(1), 'Increase phi')
  sv.register_a_shortcut('p', lambda: change_phi(-1), 'Decrease phi')

  def sig_gen(_sv: SignalViewer):
    fs, T, phi, omega = [_sv.get(k) for k in ('fs', 'T', 'phi', 'omega')]
    t = np.arange(T * fs + 1) / fs
    x = np.sin(2 * np.pi * omega * (t + phi))

    _sv.pictor.title_suffix = f'y = sin(2*pi*{omega:.1f}*(t+{phi:.3f}))'
    return x, t

  p.objects = [sig_gen]
  p.show()
