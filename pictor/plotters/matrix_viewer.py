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
from ..xomics.stat_analyzers import calc_CI
from .plotter_base import Plotter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import numpy as np



class MatrixViewer(Plotter):

  def __init__(self, matrices, row_keys, col_keys, pictor=None, values=None):
    # Call parent's constructor
    super(MatrixViewer, self).__init__(self.matshow, pictor)

    # Specific attributes
    self.matrices = matrices
    self.values = values
    self.row_keys = row_keys
    self.col_keys = col_keys

    # Settable attributes
    self.new_settable_attr('color_bar', True, bool, 'Color bar')
    self.new_settable_attr('cmap', 'Blues', str, 'Color map')
    self.new_settable_attr('title', True, bool, 'Whether to show title')
    self.new_settable_attr('digit', 2, int, 'Digits after decimal point')

    self.new_settable_attr('auto_fcolor', True, bool, 'Auto flip font color')
    self.new_settable_attr('wt', 0.7, float, 'White font threshold')
    self.new_settable_attr('ci', False, bool, 'Option to show CI95')
    self.new_settable_attr('std', False, bool, 'Option to show STD')
    self.new_settable_attr('fontsize', 8, int, 'Font size')

    self.new_settable_attr('hide_zero', True, bool, 'Option to hide zeros')


  def matshow(self, ax: plt.Axes, x: str, fig: plt.Figure):
    matrix = self.matrices[x]

    im = ax.matshow(matrix, cmap=self.get('cmap'))

    d = self.get('digit')
    for i in range(matrix.shape[0]):
      for j in range(matrix.shape[1]):
        # Hide zeros if required
        if self.get('hide_zero') and matrix[i, j] == 0: continue

        # Choose font color
        color = '#000'
        if self.get('auto_fcolor'):
          wt = matrix.min() + (matrix.max() - matrix.min()) * self.get('wt')
          color = '#FFF' if matrix[i, j] > wt else '#000'

        text = f'{matrix[i, j]:.{d}f}'

        if self.get('std') and self.values is not None:
          mu = matrix[i, j]
          values = self.values[x][i][j]
          assert np.mean(values) == mu
          text += f'\n ±{np.std(values):.{d}f}'

        if self.get('ci') and not self.get('std') and self.values is not None:
          mu = matrix[i, j]
          values = self.values[x][i][j]
          assert np.mean(values) == mu
          l, h = calc_CI(values, alpha=0.95, key=x)
          text += f'\n [{l:.{d}f},{h:.{d}f}]'

        ax.text(j, i, text, ha='center', va='center',
                fontsize=self.get('fontsize'), color=color)

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
    self.register_a_shortcut('I', lambda: self.flip('ci'), 'Turn on/off ci')
    self.register_a_shortcut('S', lambda: self.flip('std'), 'Turn on/off std')


  @classmethod
  def show_matrices(cls, matrix_dict, row_keys, col_keys, fig_size=(5, 5),
                    values=None, cmap='Blues'):
    from pictor.pictor import Pictor

    p = Pictor('Matrix Viewer', figure_size=fig_size)
    p.objects = list(matrix_dict.keys())

    mv = cls(matrix_dict, row_keys, col_keys, pictor=p, values=values)
    mv.set('cmap', cmap, verbose=False)
    p.add_plotter(mv)

    p.show()


  def multi_plot(self, indices: str='*', rows: int = 2,
                 hsize: int = 12, vsize: int = 6):
    import matplotlib.pyplot as plt

    if indices == '*': indices = f'1-{len(self.matrices)}'

    if '-' in indices:
      start, end = map(int, indices.split('-'))
      indices = list(range(start, end + 1))
    else: indices = list(map(int, indices.split(',')))

    N = len(indices)
    cols = (N + 1) // rows if rows > 1 else N
    fig, axes = plt.subplots(rows, cols, figsize=(hsize, vsize))
    for i, ax in zip(indices, axes.flatten()):
      x = self.pictor.objects[i - 1]
      self.matshow(ax, x, fig)

    plt.tight_layout()
    plt.show()
  mp = multi_plot

