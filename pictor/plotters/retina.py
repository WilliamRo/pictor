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
from .plotter_base import Plotter

import matplotlib.pyplot as plt
import numpy as np



class Retina(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super(Retina, self).__init__(self.imshow, pictor)

    # Specific attributes
    self.preprocessor = None

    # Settable attributes
    self.new_settable_attr('color_bar', False, bool, 'Color bar')
    self.new_settable_attr('k_space', False, bool, 'Whether to show k-space')
    self.new_settable_attr('log', False, bool, 'Use log-scale in k-space')
    self.new_settable_attr('alpha', None, float, 'Alpha value')
    self.new_settable_attr('vmin', None, float, 'Min value')
    self.new_settable_attr('vmax', None, float, 'Max value')
    self.new_settable_attr('cmap', None, float, 'Color map')
    self.new_settable_attr('interpolation', None, str, 'Interpolation method')


  @staticmethod
  def _check_image(x: np.ndarray):
    if len(x.shape) == 3 and x.shape[2] == 1: return x[:, :, 0]
    return x


  def imshow(self, ax: plt.Axes, x: np.ndarray, fig: plt.Figure):
    # Clear axes before drawing, and hide axes
    ax.set_axis_off()

    # If x is not provided
    if x is None:
      ax.text(0.5, 0.5, 'No image found', ha='center', va='center')
      return
    x = self._check_image(x)

    # Process x if preprocessor is provided
    if callable(self.preprocessor):
      x: np.ndarray = self.preprocessor(x)

    # Do 2D DFT if required
    if self.get('k_space'):
      x: np.ndarray = np.abs(np.fft.fftshift(np.fft.fft2(x)))
      if self.get('log'): x: np.ndarray = np.log(x + 1e-10)

    # Show image
    im = ax.imshow(x, cmap=self.get('cmap'), alpha=self.get('alpha'),
                   interpolation=self.get('interpolation'),
                   vmin=self.get('vmin'), vmax=self.get('vmax'))

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)

    # TODO: show title if provided


  def register_shortcuts(self):
    self.register_a_shortcut(
      'C', lambda: self.flip('color_bar'), 'Turn on/off color bar')
    self.register_a_shortcut(
      'F', lambda: self.flip('k_space'), 'Turn on/off k-space view')
    self.register_a_shortcut(
      'L', lambda: self.flip('log'),
      'Turn on/off log scale in k-space view')


