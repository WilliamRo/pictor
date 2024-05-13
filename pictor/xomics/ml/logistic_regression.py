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
# ====-====================================================================-====
from pictor.xomics.ml.ml_engine import MLEngine
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

import numpy as np



class LogisticRegression(MLEngine):

  SK_CLASS = SKLogisticRegression

  DEFAULT_HP_SPACE = {
    'penalty': [
      # 'l1',   # l1_ratio = 1 is equivalent to using penalty='l1'
      # 'l2',   # l1_ratio = 0 is equivalent to using penalty='l2'
      'elasticnet',
      # 'none'
    ],
    'solver': [
      # 'newton-cg',   #
      # 'lbfgs',    # analogue of the Newton’s Method
      # 'liblinear',
      # 'sag',      # support only L2 penalization
      'saga',       # best choice by default
    ],
    'C': np.logspace(-1, 4, 6),
    'l1_ratio': np.linspace(0, 1, 5),
  }

  DEFAULT_HP_MODEL_INIT_KWARGS = {'tol': 1e-2}
