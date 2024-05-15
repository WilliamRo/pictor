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
def get_model_class(key):
  from pictor.xomics.ml.logistic_regression import LogisticRegression
  from pictor.xomics.ml.support_vector_machine import SupportVectorMachine
  from pictor.xomics.ml.decision_tree import DecisionTree
  from pictor.xomics.ml.random_forest import RandomForestClassifier

  try:
    from pictor.xomics.ml.xgboost import XGBClassifier
  except:
    XGBClassifier = None

  ModelClass = {
    'lr': LogisticRegression,
    'svm': SupportVectorMachine,
    'dt': DecisionTree,
    'rf': RandomForestClassifier,
    'xgb': XGBClassifier,
  }[key]

  return ModelClass
