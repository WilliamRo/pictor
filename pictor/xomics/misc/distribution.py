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
import numpy as np



def remove_outliers(data: np.ndarray, alpha=1.5):
  """Remove outliers from data with its IQR boundary
     [q1-alpha*iqr, q3+alpha*iqr]. Common non-outliers are kept.

  Euclidean distance of each sample (each row) is not considered since
  raw data may have different scale.
  """
  data = np.array(data)
  single_sequence = len(data.shape) == 1
  if single_sequence: data = data[:, np.newaxis]

  for i in range(data.shape[1]):
    x = data[:, i]
    q1, q3 = np.percentile(x, [25, 75])
    low, high = q1 - alpha * (q3 - q1), q3 + alpha * (q3 - q1)
    mask = (x >= low) & (x <= high)
    data = data[mask]

  if single_sequence: return data[:, 0]
  return data


def remove_outliers_for_list(*data_list, alpha=1.5):
  data = np.stack(data_list, axis=-1)
  data = remove_outliers(data, alpha)
  return [data[:, i] for i in range(data.shape[1])]



if __name__ == '__main__':
  a = np.arange(10)
  b = remove_outliers(a)
  print(b)
