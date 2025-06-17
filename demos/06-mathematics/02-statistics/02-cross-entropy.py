import matplotlib.pyplot as plt
import numpy as np

from pictor import Pictor, Plotter
from scipy.stats import norm



class CrossEntropyDemo(Plotter):
  xmax = 8
  mu_p = 0
  sigma_p = 1
  sigma_q = 1
  n_samples = 10001

  def __init__(self):
    super().__init__(self.plot)


  @property
  def mu_q_list(self): return self.pictor.objects

  @Plotter.property()
  def x_list(self):
    return np.linspace(-self.xmax, self.xmax, num=self.n_samples)

  @Plotter.property()
  def p(self): return norm.pdf(self.x_list, loc=self.mu_p, scale=self.sigma_p)

  @Plotter.property()
  def q_dict(self):
    q_dict = {}

    for mu in self.mu_q_list:
      q = norm.pdf(self.x_list, loc=mu, scale=self.sigma_q)
      q_dict[mu] = q

    return q_dict

  @Plotter.property()
  def numerical_cross_entropy_dict(self):
    cross_entropy_dict = {}
    dx = self.x_list[1] - self.x_list[0]

    for mu in self.mu_q_list:
      q = self.q_dict[mu]
      # Calculate cross-entropy
      ce = -np.sum(self.p * np.log(q + 1e-300)) * dx
      cross_entropy_dict[mu] = ce

    return cross_entropy_dict


  def plot(self, x, fig: plt.Figure):
    mu = x

    # (1) Draw distribution on the first row
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(self.x_list, self.p, label=f'$N(\mu_p=0, \sigma_p=1)$')
    ax1.plot(self.x_list, self.q_dict[mu], label=f'$N(\mu_q={mu:.1f}, \sigma_q=1)$')

    ax1.set_xlim([-8, 8])
    ax1.set_ylim([0, None])
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper right')

    ax1.set_title(f'Two Gaussian distributions')
    ax1.grid(True)

    # (2) Draw entropy list
    # Get cross-entropy list
    entropy_list = [self.numerical_cross_entropy_dict[m]
                    for m in self.mu_q_list]
    ce = self.numerical_cross_entropy_dict[mu]

    ax2 = fig.add_subplot(2, 1, 2)

    ax2.plot(self.mu_q_list, entropy_list)
    ax2.plot(mu, ce, 'ro')

    ax2.set_xlabel('$\mu_q$')
    ax2.set_ylabel('Cross-Entropy')
    ax2.grid(True)

    ax2.set_title(f'Cross-entropy $H(p, q)=-\int p(x)\log q(x) dx={ce:.2f}$')

    # (-1) Finalize
    fig.tight_layout()



if __name__ == '__main__':
  p = Pictor(title='Entropy', figure_size=(10, 6))

  ce = CrossEntropyDemo()
  p.add_plotter(ce)
  p.objects = np.linspace(-5, 5, 51)

  p.show()
