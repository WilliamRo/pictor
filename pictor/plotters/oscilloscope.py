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
# ===-==================================================================-=======
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .plotter_base import Plotter
from ..objects.digital_signal import DigitalSignal

import matplotlib.pyplot as plt
import numpy as np



class Oscilloscope(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super(Oscilloscope, self).__init__(self.show_signal, pictor)

    # Specific attributes


  def show_signal(self, x: np.ndarray, ax: plt.Axes):
    if x is None: return

    ax.plot(x)