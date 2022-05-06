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
# ====-==================================================================-======
from roma import Nomear
from typing import Callable

import inspect



class Plotter(Nomear):

  def __init__(self, func: Callable, pictor):
    from ..pictor import Pictor
    self.func: Callable = func
    self.pictor: Pictor = pictor

  # region: Properties

  @property
  def argument_keys(self):
    return list(inspect.signature(self.func).parameters.keys())

  # endregion: Properties

  # region: MISC

  def __call__(self):
    # Try to get key-word arguments for method according to its signature
    kwargs = {}
    for k in self.argument_keys:
      if k in ('obj', 'x'):
        kwargs[k] = self.pictor.get_element(self.pictor.Keys.OBJECTS)
      elif k in ('canvas', ):
        kwargs[k] = self.pictor.canvas
      elif k in ('fig', 'figure'):
        kwargs[k] = self.pictor.canvas.figure
      elif k in ('ax', 'axes', 'ax2d', 'axes2d'):
        kwargs[k] = self.pictor.canvas.axes2D
      elif k in ('ax3d', 'axes3d'):
        kwargs[k] = self.pictor.canvas.axes3D
      elif hasattr(self, k):
        kwargs[k] = getattr(self, k)

    # Call function
    self.func(**kwargs)

  # endregion: MISC
