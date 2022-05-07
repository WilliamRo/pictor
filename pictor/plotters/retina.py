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
# ===-=======================================================================-==
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional
from .plotter_base import Plotter

import matplotlib.pyplot as plt
import numpy as np



class Retina(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super(Retina, self).__init__(self.imshow, pictor)

    # Specific attributes
    self.color_bar: bool = False
    self.show_k_space: bool = False
    self.k_space_log: bool = False
    self.vmin = None
    self.vmax = None
    self.cmap = None
    self.interpolation = None
    self.alpha = None

    self.preprocessor = None


  def imshow(self, ax: plt.Axes, x: np.ndarray):
    # Clear axes before drawing, and hide axes
    ax.cla()
    ax.set_axis_off()

    # If x is not provided
    if x is None:
      ax.text(0.5, 0.5, 'No image found', ha='center', va='center')
      return

    # Process x if preprocessor is provided
    if callable(self.preprocessor):
      x: np.ndarray = self.preprocessor(x)

    # Do 2D DFT if required
    if self.show_k_space:
      x: np.ndarray = np.abs(np.fft.fftshift(np.fft.fft2(x)))
      if self.k_space_log: x: np.ndarray = np.log(x)

    # Show image
    im = ax.imshow(x, cmap=self.cmap, interpolation=self.interpolation,
                   alpha=self.alpha, vmin=self.vmin, vmax=self.vmax)

    # Show color bar if required
    if self.color_bar:
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      plt.colorbar(im, cax=cax)

    # TODO: show title if provided


