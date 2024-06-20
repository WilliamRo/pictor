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
# ====-=================================================================-=======
from .dr_engine import DREngine
from pictor.xomics.omix import Omix



class PCA(DREngine):
  """Principal Component Analysis (PCA) for dimension reduction."""

  TYPE = DREngine.Types.Transformer

  def _get_n_components(self, **kwargs):
    if 'n_components' not in kwargs:
      return kwargs.get('k', self.Defaults.K)
    else:
      return kwargs.get('n_components')

  def _fit_reducer(self, omix: Omix, **kwargs):
    from sklearn.decomposition import PCA

    # (1) Get configs
    n_components = self._get_n_components(**kwargs)

    # (2) Create and fit reducer
    model = PCA(n_components=n_components)
    model.fit(omix.features)

    # (3) Return reducer
    return model


  def _reduce_dimension(self, omix: Omix, **kwargs) -> Omix:
    from sklearn.decomposition import PCA

    # (1) Get configs
    n_components = self._get_n_components(**kwargs)

    # (2) Transform features, return omix
    pca: PCA  = self.reducer

    features = pca.transform(omix.features)
    feature_labels = [f'PC-{i + 1}' for i in range(n_components)]

    return omix.duplicate(features=features, feature_labels=feature_labels)
