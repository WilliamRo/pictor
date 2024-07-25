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
"""`dr` is short for `dimensionality reduction`"""
from .dr_engine import DREngine



def get_reducer_class(key):
  from pictor.xomics.ml.dr.pca import PCA
  from pictor.xomics.ml.dr.lasso import LASSO
  from pictor.xomics.ml.dr.mrmr import MRMR
  from pictor.xomics.ml.dr.indices import Indices
  from pictor.xomics.ml.dr.pval import PVAL
  from pictor.xomics.ml.dr.rfe import RFE

  ReducerClass = {
    'pca': PCA,
    'lasso': LASSO,
    'indices': Indices,
    '*': Indices,
    'mrmr': MRMR,
    'rfe': RFE,
    'sig': PVAL,
    'pval': PVAL,
  }[key.lower()]

  return ReducerClass


abbreviation_dict = {}

