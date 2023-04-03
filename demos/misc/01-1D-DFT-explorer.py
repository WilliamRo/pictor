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
import numpy as np

from pictor.plotters import SignalViewer



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
  sign = '+' if phi >= 0 else ''
  _sv.pictor.title_suffix = f'y = sin(2*pi*{omega:.1f}*(t{sign}{phi:.3f}))'
  return x, t

p.objects = [sig_gen]
p.show()