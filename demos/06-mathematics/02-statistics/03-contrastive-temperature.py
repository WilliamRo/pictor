import matplotlib.pyplot as plt
import numpy as np

from pictor import Pictor, Plotter



class Temperature(Plotter):
  n_classes = 10

  def __init__(self, confidence=0.4):
    super().__init__(self.plot)

    self.confidence = confidence


  @property
  def index(self):
    return self.n_classes // 2


  @property
  def logits(self):
    c_true = self.confidence
    l_true = np.log(c_true)
    c_other = (1 - c_true) / (self.n_classes - 1)
    l_other = np.log(c_other)
    logits = np.array([l_other] * self.n_classes)
    logits[self.index] = l_true
    return logits


  def plot(self, x, ax: plt.Axes):
    tau = x
    logits = self.logits / tau
    q = np.exp(logits) / np.sum(np.exp(logits))
    # Cross entropy
    ce1 = -np.log(q[self.index])
    ce2 = -np.log(q[0])

    ticks = [f'Class {i+1}' for i in range(self.n_classes)]
    ax.bar(ticks, q, color='dodgerblue')
    ax.set_ylim([0, 1])
    ax.set_title(fr'Distribution ($\tau={x:.3f}$) loss={ce1:.3f} (GT={self.index + 1}) / {ce2:.3f} (GT $\ne$ {self.index + 1})')
    ax.grid(True)



if __name__ == '__main__':
  p = Pictor(title='Contrastive Temperature', figure_size=(10, 6))

  td = Temperature()
  p.add_plotter(td)
  p.objects = np.logspace(-1.5, 0, 20)[::-1]

  p.show()
