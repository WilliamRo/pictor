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
# ======-=========================================================-=============
from pictor.plotters import Oscilloscope

import matplotlib.pyplot as plt
import numpy as np



class IQScope(Oscilloscope):

  def __init__(self, default_win_size=None, y_ticks=False):
    # Call parent's constructor
    super(IQScope, self).__init__(default_win_size=default_win_size,
                                  y_ticks=y_ticks)


  def show_signal(self, x: np.ndarray, fig: plt.Figure, i: int):
    # Clear figure
    fig.clear()

    # If x is not provided
    if x is None:
      self.show_text('No signal found', fig=fig)
      return

    # Get a Scrolling object based on input x
    s = self._get_scroll(x, i)
    self._selected_signal = s

    # Get channels [(name, x, y)]
    channels = s.get_channels('*')

    # Create subplots
    height_ratios = [1, 1]
    if self.get('bar'): height_ratios.append(sum(height_ratios) / 10)
    axs = fig.subplots(len(height_ratios), 1,
                       gridspec_kw={'height_ratios': height_ratios})

    # Plot signals in time domain
    self._plot_channels_in_one_plot(axs[0], channels, title=s.label)

    # Plot signals in frequency domain
    self._plot_frequency(axs[1], channels)

    # Show scroll-bar if necessary
    if self.get('bar'): self._outline_bar(axs[-1], s)


  def _plot_channels_in_one_plot(self, ax: plt.Axes, channels: list, title):
    for name, x, y in channels: ax.plot(x, y)

    ax.set_xlim(min(x), max(x))
    ax.set_ylabel('Time Domain')

    if title is not None: ax.set_title(title)
    ax.legend([name for name, _, _ in channels])

    if not self.get('y_ticks'): ax.set_yticklabels([])


  def _plot_frequency(self, ax: plt.Axes, channels: list):
    # Sanity check
    assert len(channels) == 2

    # Create complex signal from I/Q components
    I, Q = [y for _, _, y in channels]
    S =  I + 1j*Q

    # Do Fourier transformation and plot
    F = np.fft.fftshift(np.fft.fft(S))
    ax.plot(20 * np.log10(np.abs(F) / 1.0))

    # Set styles
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel('Frequency Domain')
    if not self.get('y_ticks'): ax.set_yticklabels([])
