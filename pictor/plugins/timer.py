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
# -============================================================================-
import numpy as np

from roma import Nomear

import time



class Timer(Nomear):

  KEY_DEFAULT = 'KEY_DEFAULT'
  MAX_TICS = 5


  @Nomear.property()
  def tic_dict(self): return {}


  @property
  def fps(self, key=KEY_DEFAULT) -> float:
    if key not in self.tic_dict: self.tic_dict[key] = []
    tics = self.tic_dict[key]

    L = len(tics)
    if L < 2: return 0
    elif tics[-1] == tics[0]: return 9999
    return (L - 1) / (tics[-1] - tics[0])


  def _tic(self, key=KEY_DEFAULT):
    if key not in self.tic_dict: self.tic_dict[key] = []
    tics = self.tic_dict[key]

    tics.append(time.time())
    if len(tics) > self.MAX_TICS: tics.pop(0)


  def _reset_tics(self, key=KEY_DEFAULT): self.tic_dict[key].clear()

