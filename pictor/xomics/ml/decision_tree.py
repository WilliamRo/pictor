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
from pictor.xomics.ml.ml_engine import MLEngine
from sklearn import tree

import numpy as np



class DecisionTree(MLEngine):

  SK_CLASS = tree.DecisionTreeClassifier

  DEFAULT_HP_SPACE = [
    {
      'criterion': ['gini', 'entropy'],
      'max_depth': [2, 4, 6, 8, 10, 12],
      'max_features': [None, 'sqrt'],
    },
  ]

  EXTRA_FIT_KWARGS = {}
