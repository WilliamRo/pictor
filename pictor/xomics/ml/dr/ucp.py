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
# ===-========================================================================-=
from .dr_engine import DREngine
from pictor.xomics.omix import Omix

import numpy as np



class UCP(DREngine):
  """Uncorrelated P-value based feature selection for dimension reduction."""

  TYPE = DREngine.Types.Selector

  class Defaults:
    K = 10          # Default number of components
    THRESHOLD = 0.9

  def _fit_reducer(self, omix: Omix, **kwargs):
    # (1) Get configs
    k = kwargs.get('k', self.Defaults.K)
    assert k > 0
    threshold = kwargs.get('threshold', self.Defaults.THRESHOLD)

    # (2) Create ranking indices
    if omix.targets_are_numerical:
      indices = np.argsort([r.f_pvalue for r in omix.OLS_reports])
    else:
      indices = np.argsort(
        [r[0][2] for r in omix.single_factor_analysis_reports])

    # (3) Calculate correlation matrix
    corr_mat = np.abs(np.corrcoef(omix.features, rowvar=False))

    # (4) Remove correlated features
    remainder = indices[:]
    uc_indices = []
    while len(remainder) > 0:
      # (4.1) Get the first feature
      uc_indices.append(remainder[0])
      remainder = remainder[1:]

      # (4.2) Remove correlated features
      remainder = [i for i in remainder
                   if max([corr_mat[i, j] for j in uc_indices]) < threshold]

    # (5) Return reducer, and indices
    return uc_indices, uc_indices[:k]

  def _reduce_dimension(self, omix: Omix, **kwargs) -> Omix:
    k = kwargs.get('k', self.Defaults.K)
    return omix.get_sub_space(self.reducer[:k], start_from_1=False)
