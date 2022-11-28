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
# ====-====================================================================-====
from roma import Nomear

import numpy as np



class LargeImage(Nomear):
  """This class is designed for visualizing large images of
     (1) 2D or 3D; (2) grey or RGB.

     Data format: (1) depth first; (2) channel last.
  """

  large_images = {}

  def __init__(self, im: np.ndarray, thumbnail_size=None):
    self.image = im
    self.thumbnail_size = thumbnail_size

    # ROI range of H/W dimensions
    self.roi_range = [[0, 1], [0, 1]]

  # region: Properties

  @property
  def dimension(self): return self.get_im_dim(self.image)

  @property
  def depth(self): return self.image.shape[0] if self.dimension == 3 else 1

  @property
  def roi_size(self):
    return (self.roi_range[0][1] - self.roi_range[0][0],
            self.roi_range[1][1] - self.roi_range[1][0])

  @property
  def image_HW(self):
    if self.dimension == 3: return self.image.shape[1:3]
    return self.image.shape[:2]

  @property
  def roi_hw_slices(self):
    return [(int(rg[0] * sz), int(rg[1] * sz))
            for rg, sz in zip(self.roi_range, self.image_HW)]

  @property
  def roi_thumbnail(self):
    if self.roi_range == [[0, 1], [0, 1]]: return self.image_thumbnail

    h_rg, w_rg = self.roi_hw_slices

    if self.dimension == 3:
      roi = self.image[:, h_rg[0]:h_rg[1], w_rg[0]:w_rg[1]]
    else: roi = self.image[h_rg[0]:h_rg[1], w_rg[0]:w_rg[1]]

    return self.shrink_im(roi, self.thumbnail_size)

  @property
  def image_thumbnail(self):
    ts = self.thumbnail_size
    key = f'image_thumbnail_{ts}'
    return self.get_from_pocket(key, initializer=lambda: self.shrink_im(
      self.image, ts))

  # endregion: Properties

  # region: Public Methods

  @classmethod
  def wrap(cls, im: np.ndarray, max_size=None):
    """Wrap an image as LargeImage"""
    key = im.ctypes.data
    if key not in cls.large_images:
      cls.large_images[key] = LargeImage(im, thumbnail_size=max_size)
    return cls.large_images[key]

  @classmethod
  def shrink_im(cls, im: np.ndarray, max_size: int):
    """Shrink image according to `max_size`.
    im: should be of shape [[D, ]H, W, ...]
    """
    if max_size is None: return im

    # Get H, W according to dimension
    dim = cls.get_im_dim(im)
    if dim == 3: H, W = im.shape[1:3]
    else: H, W = im.shape[:2]

    if max(H, W) <= max_size: return im
    if H > W:
      h = max_size
      w = int(W / H * h)
    else:
      w = max_size
      h = int(H / W * w)

    # To shrink an image, it will generally look best with #INTER_AREA
    import cv2
    resize = lambda x: cv2.resize(x, dsize=(h, w), interpolation=cv2.INTER_AREA)
    if dim == 2: return resize(im)
    # For 3-D stack
    return np.stack([resize(im2d) for im2d in im], axis=0)

  @staticmethod
  def get_im_dim(im: np.ndarray):
    # For grey images
    dim_modifier = 0
    # For RGB images and grey images with last dim equal to 1
    if im.shape[-1] in (1, 3): dim_modifier = -1
    dim = len(im.shape) + dim_modifier
    assert dim in (2, 3)
    return dim

  def set_roi(self, h_range=(0, 1), w_range=(0, 1)):
    self.roi_range = [
      self._check_range(x_range) for x_range in (h_range, w_range)]

  def move_roi(self, h_step=0, w_step=0):
    new_roi_range = [[], []]
    for i, (step, size) in enumerate(zip((h_step, w_step), self.roi_size)):
      delta = step * size
      range_after_move = self._check_range(
        [x + delta for x in self.roi_range[i]])

      # Calculate true delta
      index = 1 if delta > 0 else 0
      delta = range_after_move[index] - self.roi_range[i][index]
      new_roi_range[i] = [x + delta for x in self.roi_range[i]]

    # Move ROI by setting roi_range
    self.set_roi(*new_roi_range)

  # endregion: Public Methods

  # region: Private Methods

  @staticmethod
  def _check_range(x_range):
    assert isinstance(x_range, (tuple, list)) and len(x_range) == 2
    assert x_range[0] < x_range[1]
    return [max(0, x_range[0]), min(1.0, x_range[1])]

  # endregion: Private Methods



