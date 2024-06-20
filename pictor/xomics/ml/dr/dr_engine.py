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
# ===-=====================================================================-====
from pictor.xomics.omix import Omix
from roma import Nomear



class DREngine(Nomear):

  class Types:
    Selector = 'selector'
    Transformer = 'transformer'

  class Keys:
    Reducer = 'DREngine::Reducer'
    SelectedIndices = 'DREngine::SelectedIndices'

  class Defaults:
    K = 10          # Default number of components

  TYPE = None

  def __init__(self, standardize: bool=True, **configs):
    self.standardize = standardize
    self.configs = configs

  # region: Properties

  @property
  def reducer(self): return self.get_from_pocket(self.Keys.Reducer)

  @property
  def selected_indices(self):
    assert self.TYPE == self.Types.Selector
    return self.get_from_pocket(self.Keys.SelectedIndices,
                                key_should_exist=True)

  # endregion: Properties

  # region: Public Methods

  def reduce_dimension(self, omix: Omix, **kwargs) -> Omix:
    # (0) Update configs
    configs = self.configs.copy()
    configs.update(kwargs)

    # (1) Standardize if required
    if self.standardize: omix = omix.standardize()

    # (2) Return dimension-reduced Omix
    return self._reduce_dimension(omix, **configs)

  def fit_reducer(self, omix: Omix, **kwargs):
    # (0) Update configs
    configs = self.configs.copy()
    configs.update(kwargs)

    # (1) Standardize if required
    if self.standardize: omix = omix.standardize()

    # (2) Fit the reducer
    if self.TYPE == self.Types.Selector:
      reducer, indices = self._fit_reducer(omix, **configs)
      assert indices is not None, 'Indices must be returned for selector.'
      self.put_into_pocket(self.Keys.SelectedIndices, indices, local=True)
    elif self.TYPE == self.Types.Transformer:
      reducer = self._fit_reducer(omix, **configs)
    else: raise TypeError(f'Invalid TYPE: {self.TYPE}')

    self.put_into_pocket(self.Keys.Reducer, reducer, local=True)

  # endregion: Public Methods

  # region: Private Methods


  # endregion: Private Methods

  # region: APIs

  def _fit_reducer(self, omix: Omix, **kwargs):
    raise NotImplementedError

  def _reduce_dimension(self, omix: Omix, **kwargs) -> Omix:
    if self.TYPE == self.Types.Selector:
      return omix.get_sub_space(self.selected_indices, start_from_1=False)
    raise NotImplementedError

  # endregion: APIs
