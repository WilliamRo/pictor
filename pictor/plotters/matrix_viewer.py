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
# ==-=======================================================================-===
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .plotter_base import Plotter

import matplotlib.pyplot as plt
import numpy as np



class MatrixViewer(Plotter):

  def __init__(self, matrices, row_keys, col_keys, pictor=None):
    # Call parent's constructor
    super(MatrixViewer, self).__init__(self.matshow, pictor)

    # Specific attributes
    self.matrices = matrices
    self.row_keys = row_keys
    self.col_keys = col_keys

    # Settable attributes
    self.new_settable_attr('color_bar', True, bool, 'Color bar')
    self.new_settable_attr('cmap', 'Blues', str, 'Color map')
    self.new_settable_attr('title', True, bool, 'Whether to show title')
    self.new_settable_attr('digit', 3, int, 'Digits after decimal point')

    self.new_settable_attr('auto_fcolor', True, bool, 'Auto flip font color')
    self.new_settable_attr('wt', 0.7, float, 'While font threshold')


  def matshow(self, ax: plt.Axes, x: str, fig: plt.Figure):
    matrix = self.matrices[x]

    im = ax.matshow(matrix, cmap=self.get('cmap'))

    d = self.get('digit')
    for i in range(matrix.shape[0]):
      for j in range(matrix.shape[1]):
        color = '#000'
        if self.get('auto_fcolor'):
          wt = matrix.min() + (matrix.max() - matrix.min()) * self.get('wt')
          color = '#FFF' if matrix[i, j] > wt else '#000'
        ax.text(j, i, f'{matrix[i, j]:.{d}f}', ha='center', va='center',
                color=color)

    if self.get('title'): ax.set_title(x)

    ax.set_xticks(range(len(self.col_keys)), self.col_keys)
    ax.set_yticks(range(len(self.row_keys)), self.row_keys)

    # Move xticks to the bottom
    ax.xaxis.set_ticks_position('bottom')

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)


  def register_shortcuts(self):
    self.register_a_shortcut(
      'T', lambda: self.flip('title'), 'Turn on/off title')
    self.register_a_shortcut(
      'C', lambda: self.flip('color_bar'), 'Turn on/off color bar')
    self.register_a_shortcut(
      'A', lambda: self.flip('auto_fcolor'), 'Turn on/off auto_fcolor')


  @classmethod
  def show_matrices(cls, matrix_dict, row_keys, col_keys, fig_size=(5, 5)):
    from pictor.pictor import Pictor

    p = Pictor('Matrix Viewer', figure_size=fig_size)
    p.objects = list(matrix_dict.keys())

    mv = cls(matrix_dict, row_keys, col_keys, pictor=p)
    p.add_plotter(mv)

    p.show()

