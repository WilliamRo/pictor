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

  def animate(self, axis: str = None, fps: float = 2, cursor_range: str = None,
              fmt: str = 'gif', path: str = None, n_tail: int = 0):
    """Export animation. The basic syntax is
    `ani [axis] [fps] [cursor_range] [format] [path] [n_tail]`

    Arguments
    ---------
    axis: Axis to traverse through, should be in ['o', 'p'] (correspoding to
          objects and plotters, respectively)
    fps: Frame per second
    cursor_range: Range of cursor, e.g., `10:20`
    fmt: Format of exported file, can be 'gif' or 'mp4'
    path: Path to save the file
    n_tail: A workaround to avoid losing last few frames when export mp4

    If `fmt` is `mp4`, ffmpeg must be installed.
    Official instruction for windows system: https://www.wikihow.com/Install-FFmpeg-on-Windows
    """
    from pictor.pictor import Pictor
    assert isinstance(self, Pictor)

    # Set function
    if axis == 'o':
      _func = self.so
      elements = self.axes[self.Keys.OBJECTS]
    elif axis == 'p':
      _func = self.sp
      elements = self.axes[self.Keys.PLOTTERS]
    else: raise ValueError('!! First parameter must be `o` or `p`')

    # Find cursor range
    if cursor_range is None: begin, end = 1, len(elements)
    else:
      if re.match('^\d+:\d+$', cursor_range) is None:
        raise ValueError('!! Illegal cursor range `{}`'.format(cursor_range))
      begin, end = [int(n) for n in cursor_range.split(':')]

    # Find path
    if fmt not in ('gif', 'mp4'):
      raise KeyError('!! `fmt` should be `gif` or `mp4`')
    if path is None:
      from tkinter import filedialog
      path = filedialog.asksaveasfilename()
    if re.match('.*\.{}$'.format(fmt), path) is None:
      path += '.{}'.format(fmt)

    # Find movie writer
    writer = None if fmt == 'gif' else animation.FFMpegWriter(fps=fps)

    # Create animation
    tgt = 'objects' if axis == 'o' else 'plotters'
    frames = list(range(begin, end))

    # TODO: directly export mp4 file will lose last few frames. Use this code
    #       block to circumvent this issue temporarily
    if fmt == 'mp4' and n_tail > 0:
      frames.extend([frames[-1] for _ in range(n_tail)])

    console.show_status(
      'Saving animation ({}[{}:{}]) ...'.format(tgt, frames[0], frames[-1]))
    def func(n):
      console.print_progress(n - begin, total=end - begin)
      _func(n)

    # This line is important when this Board is not shown
    self.refresh()

    ani = animation.FuncAnimation(
      self.canvas.figure, func, frames=frames, interval=1000 / fps)
    ani.save(path, writer=writer)
    console.show_status('Animation saved to `{}`.'.format(path))

  # Abbreviation
  ani = animate
