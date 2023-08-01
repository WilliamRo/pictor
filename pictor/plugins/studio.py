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
# ===-=========================================================================-
from roma import console

import matplotlib.animation as animation
import re



class Studio(object):
  """This plugin contains functions for exporting animations created from
  canvas"""

  # region: Animate

  def animate(self, fps: float = 2, scripts: str = None,
              cursor_range: str = None, fmt: str = 'gif', path: str = None,
              n_tail: int = 0):
    """Export animation. The basic syntax is
     `ani [fps] [scripts] [cursor_range] [format] [path] [n_tail]`

    Examples
    --------
      ani 5
      ani 8 p 10:20 path=/home/william/gifs
      ani 10 fmt=mp4 n_tail=10

    Arguments
    ---------
    fps: Frame per second
    scripts: Scripts to play, could be
             (1) a list/tuple of callable function or
             (2) one of 'o' or 'p', corresponding to objects and plotters,
                 respectively, to traverse through
    cursor_range: Range of cursor, e.g., `10:20`
    fmt: Format of exported file, can be 'gif' or 'mp4'
    path: Path to save the file
    n_tail: A workaround to avoid losing last few frames when export mp4

    If `fmt` is `mp4`, ffmpeg must be installed.
    Official instruction for windows system:
    https://www.wikihow.com/Install-FFmpeg-on-Windows
    """
    from pictor.pictor import Pictor
    assert isinstance(self, Pictor)

    # -------------------------------------------------------------------------
    #  Set function and cursor
    # -------------------------------------------------------------------------
    tgt = 'scripts'
    # Set default scripts if not provided
    if scripts is None:
      if len(self.objects) > 1: scripts = 'o'
      else: scripts = 'p'
    if scripts in ('o', 'p'):
      _func, elements, tgt = {
        'o': (self.so, self.objects, 'objects'),
        'p': (self.sp, self.plotters, 'plotters')}[scripts]
      scripts = [lambda _i=i: _func(_i + 1) for i in range(len(elements))]

    # Make sure scripts is a list/tuple of callable functions
    for f in scripts:
      if not callable(f):
        raise AssertionError('!! Elements in scripts should be callable.')

    # Find cursor range
    if cursor_range is None: begin, end = 1, len(scripts)
    else:
      if re.match('^\d+:\d+$', cursor_range) is None:
        raise ValueError('!! Illegal cursor range `{}`'.format(cursor_range))
      begin, end = [int(n) for n in cursor_range.split(':')]

    # Create function
    def func(i):
      p = i - 1 if begin <= i <= end + 1 else end
      console.print_progress(p, len(scripts))
      # Negative indices are for creating tails when `fmt` is 'mp4'
      if i < 0: return
      # Call the i-th script, here scripts = [s_1, s_2, s_3, ...]
      scripts[i - 1]()

    # Create frames (arguments passed to `func`)
    frames = list(range(begin, end + 1))
    console.show_status(
      'Saving animation ({}[{}:{}]) ...'.format(tgt, frames[0], frames[-1]))

    # TODO: directly export mp4 file will lose last few frames. Use this code
    #       block to circumvent this issue temporarily
    if fmt == 'mp4' and n_tail > 0: frames.extend([-1] * n_tail)

    # -------------------------------------------------------------------------
    #  Find path to save
    # -------------------------------------------------------------------------
    if fmt not in ('gif', 'mp4'):
      raise KeyError('!! `fmt` should be `gif` or `mp4`')
    if path is None:
      from tkinter import filedialog
      path = filedialog.asksaveasfilename()
    if re.match('.*\.{}$'.format(fmt), path) is None:
      path += '.{}'.format(fmt)

    # -------------------------------------------------------------------------
    #  Create animation
    # -------------------------------------------------------------------------
    # This line is important when this Board is not shown
    self.refresh()

    # TODO: BUG: after ani.save the animation will loop
    #            if repeat=False is not set
    # Ref: https://discourse.matplotlib.org/t/how-to-prevent-funcanimation-looping-a-single-time-after-save/21680/3
    ani = animation.FuncAnimation(
      self.canvas.figure, func, frames=frames, interval=1000 / fps,
      repeat=False)

    # Save animation using writer
    writer = None if fmt == 'gif' else animation.FFMpegWriter(fps=fps)

    ani.save(path, writer=writer)

    console.show_status('Animation saved to `{}`.'.format(path))

  # Abbreviation
  ani = animate

  # endregion: Animate
