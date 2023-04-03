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
# ======-=======================================================-===============
"""References: https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf"""
import matplotlib.pyplot as plt
import numpy as np

from pictor import Pictor
from pictor.plotters.plotter_base import Plotter



# Sampling frequency
fs = 100.0
# Time interval [0, T]
T = 2
# Generate ticks
t = np.arange(T * fs + 1) / fs

signals = []
# Signal 1
period = 1
w = 2 * np.pi / period
s1 = np.sin(w * t)
signals.append(s1)

# Signal 2
s2 = np.sin(w * (t + 0.7))
signals.append(s2)

def plot_time(x: np.ndarray, ax: plt.Axes):
  # Plot signal
  ax.plot(t, x)

  # Set styles
  ax.set_xlim(0, T)
  ax.set_xlabel('Time')

def plot_frequency(self, x: np.ndarray, ax: plt.Axes):
  # Get frequency ticks
  freq = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / fs))

  # Perform 1D DFT
  X = np.fft.fftshift(np.fft.fft(x))

  # Plot spectrum
  ref = 1
  ax.plot(freq, 20 * np.log10(np.abs(X) / ref))

  # Set styles
  ax.set_xlim(0, freq[-1])

  ymin, ymax = self.get('ymin'), self.get('ymax')
  ax.set_ylim(ymin, ymax)
  ax.set_xlabel('Frequency (Hz)')
  ax.set_ylabel('Decibels')

# Visualize signal using Pictor
p = Pictor(title='1D-DFT Explorer', figure_size=(8, 4))
p.objects = signals
p.add_plotter(plot_time)

pf = Plotter(plot_frequency)
pf.func = lambda x, ax: plot_frequency(pf, x, ax)
pf.new_settable_attr('ymin', None, float, 'Minimum dB to display')
pf.new_settable_attr('ymax', None, float, 'Maximum dB to display')
p.add_plotter(pf)

p.show()







