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
# ====-========================================================================-
from collections import OrderedDict

import numpy as np



def single_factor_analysis(groups: list):
  reports = []
  N = len(groups)
  for i in range(N):
    for j in range(i + 1, N):
      group1, group2 = groups[i], groups[j]
      p_val, method = auto_dual_test(group1, group2, return_detail=True)
      reports.append((i, j, p_val, method))

  return sorted(reports, key=lambda x: x[2], reverse=True)


def auto_dual_test(group1, group2, return_detail=False):
  from scipy import stats

  # (1) Test normality
  normality = all([test_normality(group) for group in (group1, group2)])

  # (2) Perform t-test or Mann-Whitney U test
  if normality:
    _, lev_p_val = stats.levene(group1, group2, center='mean')
    equal_var = lev_p_val > 0.05
    # Two-sample t-test
    _, p_val = stats.ttest_ind(group1, group2, equal_var=equal_var)
  else:
    # Mann-Whitney U test
    _, p_val = stats.mannwhitneyu(group1, group2)

  if return_detail: return p_val, ('t' if normality else 'u')
  return p_val


def test_normality(x, method='auto', verbose=False, return_details=False,
                   significance_level=0.05):
  from scipy import stats

  results = OrderedDict()
  method = method.lower()

  if method in ('shapiro-wilk', 'sw', 'auto'):
    statistic, p_val = stats.shapiro(x)
    if verbose: print('.. P-value of Shapiro-Wilk: {:.4f}'.format(p_val))
    results['sw'] = (statistic, p_val, p_val >= significance_level)
  if method in ('kolmogorov-smirnov', 'ks', 'auto'):
    statistic, p_val = stats.kstest(x, 'norm')
    if verbose: print('.. P-value of Kolmogorov-Smirnov: {:.4f}'.format(p_val))
    results['ks'] = (statistic, p_val, p_val >= significance_level)
  if method in ('anderson-darling', 'ad', 'auto'):
    ad = stats.anderson(x, dist='norm')
    if verbose: print('.. P-value of Anderson-Darling: {:.4f}'.format(p_val))
    critical_value = ad.critical_values[ad.significance_level ==
      significance_level * 100][0]
    results['ad'] = (ad.statistic, None, ad.statistic < critical_value)

  if len(results) == 0: raise ValueError('Invalid method: {}'.format(method))

  test_result = np.average([result[2] for _, result in results.items()]) > 0.5
  if verbose: print('>> Normality test result: {}'.format(test_result))

  if return_details: return results
  return test_result



if __name__ == '__main__':
  from scipy import stats

  repeats = 1000
  print(f'>> Testing normality, n_repeat={repeats}')
  print('-' * 79)
  for N in (10, 20, 50, 100, 200):
    acc = sum([test_normality(np.random.randn(N))
               for _ in range(repeats)]) / repeats
    print(f'N = {N}, Norm Accuracy = {acc * 100:.1f}%')

    acc = sum([not test_normality(np.random.rand(N))
               for _ in range(repeats)]) / repeats
    print(f'N = {N}, ~Norm Accuracy = {acc * 100:.1f}%')
    print('-' * 79)



