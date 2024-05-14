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
from pictor.xomics.omix import Omix
from roma import console
from roma import Nomear
from roma import io

import os
import warnings



class RepeatEvaluator(Nomear):
  """Repeat evaluation for machine learning models.

  Report format:
  rer = [

  ]
  """

  def __init__(self, omix: Omix, ignore_warnings=False):
    # 0. Ignore warnings if required
    if ignore_warnings:
      warnings.simplefilter('ignore')
      os.environ["PYTHONWARNINGS"] = "ignore"
      console.show_status('Warning Ignored.', prompt='[RER] >>')

    self.omix = omix

  # region: Properties

  @Nomear.property(local=True)
  def sub_space_omices(self): return []

  @Nomear.property()
  def learners(self): return []

  # endregion: Properties

  # region: Feature Selection

  def create_sub_feature_space(self, method: str, repeats=1,
                               show_progress=0, **kwargs):
    method = method.lower()
    prompt = '[FEATURE SELECTION] >>'

    if method == 'pca': assert repeats == 1, "Repeat PCA makes no sense."

    if show_progress: console.show_status(
      f'Creating sub-feature space using `{method}` ...', prompt=prompt)

    for i in range(repeats):
      if show_progress: console.print_progress(i, repeats)
      omix_sub = self.omix.select_features(method, **kwargs)
      self.sub_space_omices.append(omix_sub)

    if show_progress: console.show_status(
      f'{repeats} sub-feature spaces created.', prompt=prompt)

  # endregion: Feature Selection

  # region: Fitting

  def

  # endregion: Fitting

  # region: Public Methods

  def eval_reduce_fit(self,
                      reduce_method: dict=None,
                      fitting_method: dict=None,
                      reduce_repeats=1,
                      fitting_repeats=1,
                      **kwargs):
    """Repeatedly evaluate a given `reduce -> fit` pipline.

    reduce_method example:
      -

    """
    # (1) Select features for `reduce_repeats` times
    reduced_omices = []
    if reduce_method is None: reduced_omices.append(self)
    else:
      pass

  # endregion: Public Methods

  # region: IO

  @staticmethod
  def load(file_path: str, verbose=True):
    return io.load_file(file_path, verbose=verbose)

  def save(self, file_path: str, verbose=True):
    if not file_path.endswith('.rer'): file_path += '.rer'
    return io.save_file(self, file_path, verbose=verbose)

  # endregion: IO
