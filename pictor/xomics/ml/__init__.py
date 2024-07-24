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
from pictor.xomics.ml.logistic_regression import LogisticRegression
from pictor.xomics.ml.support_vector_machine import SupportVectorMachine
from pictor.xomics.ml.decision_tree import DecisionTree
from pictor.xomics.ml.random_forest import RandomForestClassifier

try:
  from pictor.xomics.ml.xgboost import XGBClassifier
except:
  print('! XGBoost is not installed.')
  XGBClassifier = None



MODEL_CLASS_DICT = {
  'lr': LogisticRegression,
  'svm': SupportVectorMachine,
  'dt': DecisionTree,
  'rf': RandomForestClassifier,
  'xgb': XGBClassifier,
}

SK_TO_OMIX_DICT = {model_class.SK_CLASS: model_class
                   for _, model_class in MODEL_CLASS_DICT.items()}

def get_model_class(key): return MODEL_CLASS_DICT[key]


abbreviation_dict = {
  'LogisticRegression': 'LR',
  'DecisionTree': 'DT',
  'SupportVectorMachine': 'SVM',
  'RandomForestClassifier': 'RF',
  'XGBClassifier': 'XGB',
}
