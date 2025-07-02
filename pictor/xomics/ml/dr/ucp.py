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
from pictor.xomics.ml.dr import DREngine
# from .dr_engine import DREngine
from pictor.xomics.omix import Omix

import numpy as np
import time



class UCP(DREngine):
  """Uncorrelated P-value based feature selection for dimension reduction."""

  TYPE = DREngine.Types.Selector

  class Defaults:
    K = 10          # Default number of components
    THRESHOLD = 0.9
    M = None

  def _fit_reducer(self, omix: Omix, **kwargs):
    # (1) Get configs
    k = kwargs.get('k', self.Defaults.K)
    assert k > 0
    threshold = kwargs.get('threshold', self.Defaults.THRESHOLD)
    m = kwargs.get('m', self.Defaults.M)

    self.dev_report(f'k = {k}, threshold = {threshold}, m = {m}')

    # (2) Create ranking indices
    tic = time.time()
    self.dev_report(f'Ranking {omix.n_features} features from {omix.n_samples} samples ...')
    if omix.targets_are_numerical:
      indices = np.argsort([r.f_pvalue for r in omix.OLS_reports])
    else:
      indices = np.argsort(
        [r[0][2] for r in omix.single_factor_analysis_reports])

    self.dev_report(f'Feature ranking took {time.time() - tic:.2f} seconds')

    # (3) Calculate correlation matrix
    tic = time.time()
    corr_mat = np.abs(np.corrcoef(omix.features, rowvar=False))
    self.dev_report(f'Correlation matrix calculated in {time.time() - tic:.2f} seconds')

    # (4) Remove correlated features
    # TODO
    from roma import console

    self.dev_report('Selecting features ...')
    remainder = indices[:]

    # This line speeds up the process significantly
    if m is not None: remainder = remainder[:m]

    uc_indices = []
    while len(remainder) > 0:
      # TODO
      # if self.dev_mode:
      #   _tic = time.time()
      #   console.print_progress(len(indices) - len(remainder), len(indices))

      version = 2
      if version == 1:
        # (4.1) Get the first feature
        uc_indices.append(remainder[0])
        remainder = remainder[1:]

        # (4.2) Remove correlated features
        remainder = [i for i in remainder
                     if max([corr_mat[i, j] for j in uc_indices]) < threshold]
      elif version == 2:
        uc_indices.append(remainder[0])
        if len(uc_indices) == k:
          break

        submat = corr_mat[remainder, :][:, uc_indices]
        max_corr = np.max(submat, axis=1)
        keep_mask = max_corr < threshold
        remainder = [remainder[k] for k in np.where(keep_mask)[0]]
      else: raise ValueError(f'Invalid version: {version}')

      # TODO
      # if self.dev_mode: console.show_status(f'{time.time() - _tic:.2f} seconds, {len(remainder)} features left',)

    self.dev_report(f'Feature selection completed.')

    # (5) Return reducer, and indices
    return uc_indices, uc_indices[:k]

  def _reduce_dimension(self, omix: Omix, **kwargs) -> Omix:
    k = kwargs.get('k', self.Defaults.K)
    return omix.get_sub_space(self.reducer[:k], start_from_1=False)



if __name__ == '__main__':
  from pictor.xomics.omix import Omix
  DREngine.enable_dev_mode()

  # Configuration
  n_samples = 100
  # n_features = 1000
  n_features = 2000

  # Test
  omix = Omix.gen_psudo_omix(n_samples, n_features)
  omix.select_features('UCP', k=10, threshold=0.9, m=500)
  # omix.show_in_explorer()

  print('Done.')
