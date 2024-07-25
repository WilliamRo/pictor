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
# ==-==========================================================================-
from .dr_engine import DREngine
from pictor.xomics.omix import Omix



class Indices(DREngine):
  """Select features by specified indices"""

  TYPE = DREngine.Types.Selector


  def _fit_reducer(self, omix: Omix, indices='*', start_from_1=False, **kwargs):
    if isinstance(indices, str):
      if indices == '*':
        indices = list(range(omix.n_features))
      elif '-' in indices:
        start, end = map(int, indices.split('-'))
        indices = list(range(start, end + 1))
      elif ',' in indices:
        indices = list(map(int, indices.split(',')))

    if start_from_1: indices = [i - 1 for i in indices]

    return None, indices

