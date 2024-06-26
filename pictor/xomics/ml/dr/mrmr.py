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
from .dr_engine import DREngine
from pictor.xomics.omix import Omix



class MRMR(DREngine):
  """Minimum Redundancy Maximum Relevance (MRMR) Dimensionality Reduction"""

  TYPE = DREngine.Types.Selector


  def _fit_reducer(self, omix: Omix, **kwargs):
    from mrmr import mrmr_classif
    import pandas as pd

    # (1) Get configs
    k = kwargs.get('k', self.Defaults.K)

    # (2) Create and fit reducer
    indices = mrmr_classif(
      X=pd.DataFrame(omix.features), y=pd.Series(omix.targets), K=k)

    # (3) Return reducer, and indices
    return 'MRMR', indices
