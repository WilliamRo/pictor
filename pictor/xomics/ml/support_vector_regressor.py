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
# ===-=========================================================================-
from pictor.xomics.ml.ml_engine import MLEngine
from sklearn import svm

import numpy as np



class SupportVectorRegressor(MLEngine):

  SK_CLASS = svm.SVR

  DEFAULT_HP_SPACE = [
    {
      'C': np.logspace(-1, 3, 5),
      'kernel': ['linear'],
    },
    {
      'C': np.logspace(-1, 3, 5),
      'kernel': ['poly', 'rbf', 'sigmoid'],
      'gamma': ['scale', 'auto'],
    },
  ]

  EXTRA_FIT_KWARGS = {'max_iter': int(1e6)}

  DEFAULT_HP_MODEL_INIT_KWARGS = {'max_iter': int(1e6)}
