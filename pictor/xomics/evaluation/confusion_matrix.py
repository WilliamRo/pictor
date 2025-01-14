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
# =-===========================================================================-
from roma import check_type
from roma import Table


import numpy as np



class ConfusionMatrix(object):
  """ Matrix Organization:
                                      Actual Classes
                             class_1    class_2  ...  class_N
                   class_1   m[0,0]     m[0,1]   ...  m[0,N-1]
                   class_2   m[1,0]     m[1,1]   ...  m[1,N]-1
      Predicted      .          .          .     .       .
       Classes       .          .          .      .      .
                     .          .          .       .     .
                   class_N  m[N-1,0]   m[N-1,1]  ... m[N-1,N-1]
  """

  def __init__(self, num_classes, class_names=None):
    self.num_classes = check_type(num_classes, int)
    self._class_names = class_names
    if class_names is not None:
      assert all([isinstance(class_names, (tuple, list)),
                  len(class_names) == num_classes])
      check_type(class_names, str)

    # Variables to be filled
    self.predictions, self.targets = None, None

    self.support, self.total = None, None
    self.confusion_matrix = None
    self.TPs, self.TNs, self.FPs, self.FNs = None, None, None, None
    self.precisions, self.recalls, self.F1s = None, None, None
    self.macro_precision, self.weighted_precision = None, None
    self.macro_recall, self.weighted_recall = None, None
    self.macro_F1, self.weighted_F1 = None, None
    self.accuracy = None
    self.missed_indices = None


  @property
  def class_names(self):
    return (['Class_{}'.format(i + 1) for i in range(self.num_classes)]
            if self._class_names is None else self._class_names)


  @staticmethod
  def calculate_PRF(TP, FP, FN):
    precision = TP / np.maximum(TP + FP, 1)
    recall = TP / np.maximum(TP + FN, 1)
    F1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    return precision, recall, F1


  def fill(self, preds, truths):
    # Sanity check
    if not isinstance(preds, np.ndarray): preds = np.array(preds)
    if not isinstance(truths, np.ndarray): truths = np.array(truths)
    preds, truths = np.ravel(preds), np.ravel(truths)
    assert all([preds.size == truths.size, preds.max() < self.num_classes,
                truths.max() < self.num_classes, ])
    total = preds.size

    self.predictions = preds
    self.targets = truths
    self.missed_indices = np.argwhere(preds != truths).ravel()

    # Initialize matrix
    cm = np.zeros(shape=[self.num_classes, self.num_classes], dtype=int)
    support = np.zeros(shape=[self.num_classes], dtype=int)
    for c in range(self.num_classes):
      mask = truths == c
      support[c] = np.sum(mask)
      c_preds = preds[mask]
      cm[c] = np.array([np.sum(c_preds==i) for i in range(self.num_classes)])
    assert np.sum(cm) == total
    cm = cm.transpose()

    # Count positives and negatives
    self.TPs = cm.diagonal()
    self.FPs = np.sum(cm, axis=1) - self.TPs
    self.FNs = np.sum(cm, axis=0) - self.TPs
    self.TNs = total - self.FPs - self.FNs - self.TPs
    assert all(self.TPs + self.FPs + self.TNs + self.FNs == np.array(
      [total] * self.num_classes))

    # Calculate performance measures for each class
    self.precisions, self.recalls, self.F1s = self.calculate_PRF(
      self.TPs, self.FPs, self.FNs)

    # Calculate overall performance measures
    values = (self.precisions, self.recalls, self.F1s)
    self.macro_precision, self.macro_recall, self.macro_F1 = [
      np.average(val) for val in values]

    self.weighted_precision, self.weighted_recall, self.weighted_F1 = [
      np.average(val, weights=support) for val in values]

    self.accuracy = np.sum(self.TPs) / total

    # Set variables
    self.total = total
    self.support = support
    self.confusion_matrix = cm


  def make_result_table(self, class_details=True, decimal=3, tab=2, margin=1,
                        groups=None):
    """Produce a sklearn style table. Format:

      Table 1: Example Table
      ------------------------------------------------------
                      Precision   Recall  F1-Score  Support
      ======================================================
       Goose              0.308    0.667    0.421         6
       Cat                0.667    0.200    0.308        10
       Dog                0.667    0.667    0.667         9
      ------------------------------------------------------
       Accuracy                             0.480        25
       Macro Avg          0.547    0.511    0.465        25
       Weighted Avg       0.581    0.480    0.464        25
      ------------------------------------------------------
    """
    names = self.class_names
    # Width are decided according to Tab.1
    table = Table(12, 9, 8, 8, 7, tab=tab, margin=margin, buffered=True)
    table.specify_format(
      None, *(['{:.' + str(decimal) + 'f}'] * 3), None, align='lrrrr')
    table.print_header('', 'Precision', 'Recall', 'F1-Score', 'Support')
    # Show performance detail for each class
    if class_details:
      for i, name in enumerate(names):
        table.print_row(name, self.precisions[i], self.recalls[i],
                        self.F1s[i], self.support[i])
      table.hline()
    # Show overall performance
    table.print_row('Accuracy', '', '', self.accuracy, self.total)
    table.print_row('Macro Avg', self.macro_precision, self.macro_recall,
                    self.macro_F1, self.total)
    table.print_row('Weighted Avg', self.weighted_precision,
                    self.weighted_recall, self.weighted_F1, self.total)
    table.hline()
    # Add group information if required
    if groups:
      for row in self.merge(groups): table.print_row(*row)
      table.hline()
    return table


  def merge(self, groups):
    assert isinstance(groups, (tuple, list))
    if not isinstance(groups[0], (tuple, list)): groups = [groups]
    results = []
    cm = self.confusion_matrix
    for indices in groups:
      assert isinstance(indices, (tuple, list))
      assert all([isinstance(i, int) and i >= 0 for i in indices])
      assert len(indices) == len(set(indices))
      n = '/'.join([self.class_names[i] for i in indices])
      ind = np.array(indices)
      TP = np.sum(cm[ind][:, ind])
      FP = np.sum(cm[ind]) - TP
      FN = np.sum(cm[:, ind]) - TP
      p, r, f = self.calculate_PRF(TP, FP, FN)
      s = np.sum(self.support[ind])
      results.append((n, p ,r, f, s))
    return results


  def make_matrix_table(self, cell_width=None):
    names = self.class_names
    ncols = len(names) + 2
    if cell_width is None: cell_width = max([len(s) for s in names])
    # Make table
    table = Table(*([cell_width] * ncols), buffered=True, margin=1, tab=2)
    table.specify_format(*([None] * ncols), align='r' * ncols)
    table.hline()
    # First line
    cells = [''] * ncols
    cells[len(names) // 2 + 2] = 'True'
    table.print_row(*cells)
    table.hline()
    # Second line
    table.print_row('', '', *names)
    table.hline()
    # Print matrix
    for i, name in enumerate(names):
      cells = ['Predicted' if i == len(names) // 2 else '', name] + [
        n for n in self.confusion_matrix[i]]
      table.print_row(*cells)
    table.hline()
    return table


  def sklearn_plot(self, **kwargs):
    """Plot confusion matrix using sklearn package"""
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    cm = self.confusion_matrix
    assert isinstance(cm, np.ndarray)
    disp = ConfusionMatrixDisplay(
      cm.transpose(), display_labels=self.class_names)
    disp.plot(cmap=plt.cm.Blues, **kwargs)
    plt.show()


  def __getitem__(self, key: str):
    """This method is only for binary classification"""
    assert len(self.TPs) == 2, 'This method is only for binary classification'

    TP, FN = self.TPs[1], self.FNs[1]
    FP, TN = self.FPs[1], self.TNs[1]
    P, N = TP + FN, FP + TN
    PP, PN = TP + FP, FN + TN

    key = key.lower()
    if key in ('tpr', 'recall', 'true positive rate', 'sensitivity'):
      return TP / np.maximum(P, 1)
    if key in ('fnr', 'miss rate', 'false negative rate'):
      return FN / np.maximum(P, 1)
    if key in ('tnr', 'specificity', 'true negative rate', 'spc', 'selectivity'):
      return TN / np.maximum(N, 1)
    if key in ('fpr', 'fall-out', 'false positive rate'):
      return FP / np.maximum(N, 1)

    if key in ('ppv', 'precision', 'positive predictive value'):
      return TP / np.maximum(PP, 1)
    if key in ('fdr', 'false discovery rate'):
      return FP / np.maximum(PP, 1)
    if key in ('for', 'false omission rate'):
      return FN / np.maximum(PN, 1)
    if key in ('npv', 'negative predictive value'):
      return TN / np.maximum(PN, 1)

    if key in ('f1', 'f1-score'):
      # TODO: the line below is not right
      # return 2 * TP / np.maximum(2 * TP + FP + FN, 1)
      return self.macro_F1
    if key in ('acc', 'accuracy'):
      return (TP + TN) / np.maximum(P + N, 1)
    if key in ('balanced accuracy', 'ba'):
      return (TP / np.maximum(P, 1) + TN / np.maximum(N, 1)) / 2

    if key in ('mcc', 'matthews correlation coefficient'):
      return (TP * TN - FP * FN) / np.maximum(
        np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 1)
    if key in ('bm', 'bookmaker informedness'):
      return self['tpr'] + self['tnr'] - 1

    raise ValueError(f'!! Unknown key `{key}` !!')



if __name__ == '__main__':
  cm = ConfusionMatrix(3, class_names=('Goose', 'Cat', 'Dog'))
  truths = [0] * 6 + [1] * 10 + [2] * 9
  preds = [0, 0, 0, 0, 1, 2,
           0, 0, 0, 0, 0, 0, 1, 1, 2, 2,
           0, 0, 0, 2, 2, 2, 2, 2, 2]
  cm.fill(preds, truths)
  print(cm.make_matrix_table())
  print(cm.make_result_table(groups=(1, 2)))
  # cm.sklearn_plot()
