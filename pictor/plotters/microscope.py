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
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pictor.objects.image.large_image import LargeImage
from pictor.plotters.plotter_base import Plotter
from roma import console



class Microscope(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super(Microscope, self).__init__(self.show_sample, pictor)

    # Buffers
    self._current_li: LargeImage = None

    # Settable attributes
    self.new_settable_attr('max_size', 1000, int, 'Maximum edge size to plot')

    self.new_settable_attr('color_bar', False, bool, 'Color bar')
    self.new_settable_attr('cmap', None, str, 'Color map')
    self.new_settable_attr('k_space', False, bool, 'Whether to show k-space')
    self.new_settable_attr('hist', False, bool, 'Whether to show histogram')
    self.new_settable_attr('log', False, bool, 'Use log-scale in k-space')
    self.new_settable_attr('vmin', None, float, 'Min value')
    self.new_settable_attr('vmax', None, float, 'Max value')
    self.new_settable_attr('vsigma', None, int, 'Sigma coefficient for smart '
                                                'value clipping')
    self.new_settable_attr('hist_margin', None, float,
                           'Value margin to clip, should be in (0, 50)')
    self.new_settable_attr('cmap', None, str, 'Color map')
    self.new_settable_attr('interpolation', None, str, 'Interpolation method')
    self.new_settable_attr('title', False, bool, 'Whether to show title')
    self.new_settable_attr('mini_map', False, bool, 'Whether to show mini-map')
    self.new_settable_attr('mini_map_size', 0.3, float, 'Size of mini-map')
    self.new_settable_attr('move_step', 0.2, float, 'Moving step')
    self.new_settable_attr('share_roi', True, bool, 'Whether to share ROI')

  # region: Properties

  @property
  def share_roi(self): return self.get('share_roi')

  @property
  def mean_std_of_this_li(self):
    li = self._current_li
    mu = self.get_from_pocket(
      f'{str(li)}-mu', initializer=lambda: np.mean(li.image))
    sigma = self.get_from_pocket(
      f'{str(li)}-sigma', initializer=lambda: np.std(li.image))
    return mu, sigma

  # endregion: Properties

  # region: Plot Methods

  def show_sample(self, ax: plt.Axes, x: np.ndarray, fig: plt.Figure, label):
    # Clear axes before drawing, and hide axes
    ax.set_axis_off()

    # If x is not provided
    if x is None:
      self.show_text('No image found', ax)
      return
    li: LargeImage = LargeImage.wrap(x)
    li.thumbnail_size = self.get('max_size')
    x = li.roi_thumbnail

    # Set x to buffer to zoom-in and zoom-out
    self._current_li = li

    # Get slice if li is 3-D
    if li.dimension == 3:
      x = x[self.pictor.get_element('DePtH')]

    # Show title if provided
    if label is not None and self.get('title'): ax.set_title(label)

    # Show histogram if required
    if self.get('hist'):
      self._show_histogram(x, ax)
      return

    # Do 2D DFT if required
    if self.get('k_space'):
      x: np.ndarray = np.abs(np.fft.fftshift(np.fft.fft2(x)))
      if self.get('log'): x: np.ndarray = np.log(x + 1e-10)

    # Get vmin and vmax
    vmin, vmax = self._get_vrange()

    # Show image
    im = ax.imshow(x, vmin=vmin, vmax=vmax, cmap=self.get('cmap'),
                   interpolation=self.get('interpolation'))

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)

    # Show mini-map if required
    if self.get('mini_map'): self._show_mini_map(li, ax)

  def _show_histogram(self, x: np.ndarray, ax: plt.Axes):
    x = np.ravel(x)
    ax.hist(x=x, bins=50)

    ax.set_axis_on()

    # Show margin if provided
    for v in self._get_hist_margin():
      ax.plot([v, v], ax.get_ylim(), color='#AAA')

  def _show_mini_map(self, li: LargeImage, ax: plt.Axes):
    # Configs
    r = self.get('mini_map_size')      # ratio
    m = 5                 # margin
    color = 'r'

    if li.dimension == 2: H, W = li.roi_thumbnail.shape[:2]
    else: H, W = li.roi_thumbnail.shape[1:3]
    # Outline
    h, w = int(r * H), int(r * W)
    rect = patches.Rectangle((m, m), w, h, edgecolor=color, facecolor='none',
                             linestyle='-', linewidth=2, alpha=0.5)
    ax.add_patch(rect)
    # ROI
    hr, wr = li.roi_range
    # anchors
    ah, aw = int(m + hr[0] * h), int(m + wr[0] * w)
    # lengths
    lh, lw = [int(s * l) for s, l in zip(li.roi_size, [h, w])]
    rect = patches.Rectangle((aw, ah), lw, lh, edgecolor='none',
                             facecolor=color, linewidth=2, alpha=0.3)
    ax.add_patch(rect)

  # endregion: Plot Methods

  # region: ROI

  def _set_range(self, r, u1, u2):
    assert 0 <= u1 < u2 <= 1

    s = u2 - u1
    d = s * (r - 1.0) / 2
    v1, v2 = u1 - d, u2 + d

    # Apply constrains
    if v1 < 0: v1, v2 = 0, v2 - v1
    if v2 > 1: v1, v2 = v1 - v2 + 1, 1

    return v1, v2

  def zoom(self, ratio: float):
    """Zoom-in when ratio < 1, or zoom-out when ratio > 1
    """
    assert isinstance(ratio, (int, float)) and ratio > 0
    if ratio == 1.0: return

    li: LargeImage = self._current_li
    li.set_roi(*[self._set_range(ratio, *li.roi_range[i]) for i in (0, 1)])
    if self.share_roi: self.sync_roi()
    self.refresh()

  def move_roi(self, h_step: float, w_step: float):
    self._current_li.move_roi(h_step, w_step)
    if self.share_roi: self.sync_roi()
    self.refresh()

  def sync_roi(self):
    for im in self.pictor.objects:
      li = LargeImage.wrap(im)
      li.set_roi(*self._current_li.roi_range)

  # endregion: ROI

  # region: Private Methods

  def _get_hist_margin(self):
    margin = self.get('hist_margin')
    if isinstance(margin, float) and 0 < margin < 50:
      init_func = lambda: np.percentile(
        self._current_li.image, (margin, 100 - margin))
      key = f'{str(self._current_li.image)}-hist-margin-{margin}'
      return self.get_from_pocket(key, initializer=init_func)
    return None, None

  def _get_vrange(self):
    # (1) Get hist margin if provided
    v_min, v_max = self._get_hist_margin()

    # (2) Get mean and sigma if necessary (Higher priority)
    vsigma = self.get('vsigma')
    if isinstance(vsigma, int) and vsigma > 0:
      mu, sigma = self.mean_std_of_this_li
      # Set v-range
      v_min, v_max = mu - vsigma * sigma, mu + vsigma * sigma

    # (3) `vmin` and `vmax` have highest priority
    if self.get('vmin') is not None: v_min = self.get('vmin')
    if self.get('vmax') is not None: v_max = self.get('vmax')
    return v_min, v_max

  # endregion: Private Methods

  # region: APIs

  def show_image_info(self):
    # Get current large image
    li = self._current_li

    # Show whole shape
    console.show_info(f'Current selected image info:')
    console.supplement(f'shape: {li.image.shape};', level=2)
    mu, sigma = self.mean_std_of_this_li
    console.supplement(f'values: {mu:.2f}+-{sigma:.2f};', level=2)

    # Show slices
    h_rg, w_rg = li.roi_hw_slices
    console.show_info(f'ROI range: [{h_rg[0]}:{h_rg[1]}, {w_rg[0]}:{w_rg[1]}]')

  # endregion: APIs

  # region: Commands

  def register_shortcuts(self):
    self.register_a_shortcut(
      'T', lambda: self.flip('title'), 'Turn on/off title')
    self.register_a_shortcut(
      'C', lambda: self.flip('color_bar'), 'Turn on/off color bar')
    self.register_a_shortcut(
      'F', lambda: self.flip('k_space'), 'Turn on/off k-space view')
    self.register_a_shortcut(
      'G', lambda: self.flip('log'),
      'Turn on/off log scale in k-space view')
    self.register_a_shortcut(
      'M', lambda: self.flip('mini_map'), 'Turn on/off mini-map')
    self.register_a_shortcut(
      'space', lambda: self.flip('hist'), 'Toggle histogram')

    # Zoom in/out
    self.register_a_shortcut('O', lambda: self.zoom(2.0), 'Zoom out')
    self.register_a_shortcut('I', lambda: self.zoom(0.5), 'Zoom in')

    # Move ROI
    self.register_a_shortcut(
      'K', lambda: self.move_roi(-self.get('move_step'), 0), 'Move to north')
    self.register_a_shortcut(
      'J', lambda: self.move_roi(self.get('move_step'), 0), 'Move to south')
    self.register_a_shortcut(
      'H', lambda: self.move_roi(0, -self.get('move_step')), 'Move to west')
    self.register_a_shortcut(
      'L', lambda: self.move_roi(0, self.get('move_step')), 'Move to east')
    self.register_a_shortcut('Tab', self.show_image_info, 'Show image size')

  def set_value(self, vmin: str = None, vmax: str = None):
    """Set minimum value and maximum value"""
    for key, value in zip(['vmin', 'vmax'], [vmin, vmax]):
      try: self.set(key, float(value))
      except: self.set(key, None)
  sv = set_value

  def set_hist_margin(self, margin=None):
    self.set('hist_margin', margin)
  shm = set_hist_margin

  # endregion: Commands

  # region: Animation

  def anid(self, fps: float, depth_interval: str = None, step: int = 1,
           fmt: str = 'gif', path: str = None, n_tail: int = 0):
    """Export microscope ROI with given depth interval. Syntax:
     `anit [fps] [depth_interval] [step] [format] [path] [n_tail]`

    Examples
    --------
      anid 5 step=10
      anid 8 10000:12000 path=/home/william/gifs
      anid 10 fmt=mp4 n_tail=10

    Arguments
    ---------
    fps: Frame per second;
    depth_interval: Depth interval to create animation. E.g., 50:100;
    step: depth sliding step.
    fmt: Format of exported file, can be 'gif' or 'mp4'.
         If `fmt` is `mp4`, ffmpeg must be installed;
    path: Path to save the file;
    n_tail: A workaround to avoid losing last few frames when export mp4.
    """
    import re

    if self._current_li.dimension == 2: raise AssertionError(
      '!! cannot generate animation for 2-D images')

    # -------------------------------------------------------------------------
    #  Create scripts
    # -------------------------------------------------------------------------
    # Check `time_interval` if provided
    if depth_interval is not None:
      if re.match('^\d+:\d+$', depth_interval) is None:
        raise ValueError('!! Illegal time interval `{}`'.format(depth_interval))
      begin, end = [int(n) for n in depth_interval.split(':')]
    else: begin, end = 1, self._current_li.image.shape[0]

    # Calculate script length and create scripts
    scripts = [lambda _i=i: self.pictor.sd(_i) for i in range(begin, end, step)]
    # -------------------------------------------------------------------------
    #  Call animate to create animation
    # -------------------------------------------------------------------------
    self.pictor.animate(fps, scripts, fmt=fmt, path=path, n_tail=n_tail)

  # endregion: Animation
