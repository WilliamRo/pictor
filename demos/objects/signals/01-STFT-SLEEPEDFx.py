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
"""
On Windows systems, use `RefreshEnv.cmd` after adding `xai_kit_data_dir`
environment variable if necessary.
"""
from freud.talos_utils.sleep_sets.sleepedfx import SleepEDFx, SleepSet
from freud.gui.freud_gui import Freud
from pictor.objects import DigitalSignal, SignalGroup

import numpy as np
import os



# (1) Load the target signal group
data_dir = os.getenv('xai_kit_data_dir')
data_dir = os.path.join(data_dir, r'xai-kit-demo\sleepedfx-1')
assert os.path.exists(data_dir)
sgs = SleepEDFx.load_as_signal_groups(data_dir, preprocess='trim')
sg = sgs[0]

# (2) Extract 30600:31400 from first EEG channel
sg.truncate(30600, 31400)
sg.digital_signals.pop(-1)

s = sg.digital_signals[0].data[:, 0]

# (3) Perform STFT
from scipy.signal import stft
f, t, Zxx = stft(s, fs=sg.digital_signals[0].sfreq)

import matplotlib.pyplot as plt
plt.imshow(np.abs(Zxx))
plt.show()

# (optional) Visualize signal
# Freud.visualize_signal_groups(sgs)

















