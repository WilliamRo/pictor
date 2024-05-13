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
# ====-===============================================================-=========
from pictor.xomics.ml.ml_engine import MLEngine
from xgboost import XGBClassifier as _XGBClassifier
# from sklearn.ensemble import GradientBoostingClassifier

import numpy as np



class XGBClassifier(MLEngine):
  """ Random forest “bagging” minimizes the variance and overfitting,
  while GBDT “boosting” minimizes the bias and underfitting.

  While xgboost used a more regularized model formalization to control
  over-fitting, which gives it better performance.

  The name xgboost, though, actually refers to the engineering goal to push
  the limit of computations resources for boosted tree algorithms.
  Which is the reason why many people use xgboost. For model,
   it might be more suitable to be called as regularized gradient boosting.
  """

  SK_CLASS = _XGBClassifier
  # SK_CLASS = GradientBoostingClassifier

  DEFAULT_HP_SPACE = [
    {
      'learning_rate': np.logspace(-4, -1, 4),
      'n_estimators': [
        # 10,
        100,
      ],
      'max_depth': [4, 6, 8, 10],
      'subsample': [0.5, 0.75, 1.0],
    },
  ]
