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
# ===-=======================================================================-==
from .dr_engine import DREngine
from pictor.xomics.omix import Omix

import numpy as np



class RFE(DREngine):
  """Recursive Feature Elimination (RFE) for feature selection."""

  TYPE = DREngine.Types.Selector


  def _fit_reducer(self, omix: Omix, **kwargs):
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVC

    # (1) Get configs
    k = kwargs.get('k', self.Defaults.K)
    # TODO: Only 'linear' kernel is supported
    kernel = kwargs.get('kernel', 'linear')

    # (2) Create and fit reducer
    estimator = SVC(kernel=kernel)
    selector = RFE(estimator, n_features_to_select=k, step=1)
    selector = selector.fit(omix.features, omix.targets)

    # (3) Return reducer, and indices
    indices = np.arange(omix.n_features)[selector.support_]
    return selector, indices
