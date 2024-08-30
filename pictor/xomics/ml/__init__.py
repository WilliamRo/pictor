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
# ======-=======================================================-===============
from pictor.xomics.ml.elastic_net import ElasticNet
from pictor.xomics.ml.linear_regression import LinearRegression
from pictor.xomics.ml.logistic_regression import LogisticRegression
from pictor.xomics.ml.support_vector_machine import SupportVectorMachine
from pictor.xomics.ml.support_vector_regressor import SupportVectorRegressor
from pictor.xomics.ml.decision_tree import DecisionTree
from pictor.xomics.ml.random_forest import RandomForestClassifier

try:
  from pictor.xomics.ml.xgboost import XGBClassifier
except:
  print('! XGBoost is not installed.')
  XGBClassifier = None



MODEL_CLASS_DICT = {
  'lr': LogisticRegression,
  'lor': LogisticRegression,
  'svm': SupportVectorMachine,
  'svc': SupportVectorMachine,
  'dt': DecisionTree,
  'rf': RandomForestClassifier,
  # 'xgb': XGBClassifier,

  'eln': ElasticNet,
  'lir': LinearRegression,
  'svr': SupportVectorRegressor,
}
if XGBClassifier is not None: MODEL_CLASS_DICT['xgb'] = XGBClassifier

SK_TO_OMIX_DICT = {model_class.SK_CLASS.__name__: model_class
                   for _, model_class in MODEL_CLASS_DICT.items()}

def get_model_class(key):
  if not isinstance(key, str): return key
  return MODEL_CLASS_DICT[key]


abbreviation_dict = {
  'ElasticNet': 'ELN',
  'LinearRegression': 'LiR',
  'LogisticRegression': 'LoR',
  'DecisionTree': 'DT',
  'SupportVectorMachine': 'SVM',
  'SupportVectorRegressor': 'SVR',
  'RandomForestClassifier': 'RF',
  'XGBClassifier': 'XGB',
}
