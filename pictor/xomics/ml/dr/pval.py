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
from .dr_engine import DREngine
from pictor.xomics.omix import Omix

import numpy as np



class PVAL(DREngine):
  """P-value based feature selection for dimension reduction."""

  TYPE = DREngine.Types.Selector


  def _fit_reducer(self, omix: Omix, **kwargs):
    # (1) Get configs
    k = kwargs.get('k', self.Defaults.K)

    # (2) Create and fit reducer
    indices = np.argsort(
      [r[0][2] for r in omix.single_factor_analysis_reports])

    # (3) Return reducer, and indices
    return indices, indices[:k]

  def _reduce_dimension(self, omix: Omix, **kwargs) -> Omix:
    k = kwargs.get('k', self.Defaults.K)
    return omix.get_sub_space(self.reducer[:k], start_from_1=False)
