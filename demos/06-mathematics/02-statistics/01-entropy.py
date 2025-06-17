import matplotlib.pyplot as plt
import numpy as np

from pictor import Pictor, Plotter
from scipy.stats import norm



class EntropyDemo(Plotter):
  """Note that for natural log, H(N(0, 1)) = 0.5*log(2\pi e) \approx 1.4189"""
  mu = 0
  n_samples = 10001
  xmax = 5

  def __init__(self):
    super().__init__(self.plot)

    self.new_settable_attr('analytical', True, bool,
                           'Whether to show analytical entropy')


  @property
  def sigma_list(self): return self.pictor.objects

  @Plotter.property()
  def x_list(self):
    return np.linspace(-self.xmax, self.xmax, num=self.n_samples)

  @Plotter.property()
  def p_dict(self):
    # This property will not be generated until called first time
    p_dict = {}

    for sigma in self.sigma_list:
      # Generate Gaussian distribution for each sigma
      # p = np.exp(-0.5 * ((self.x_list - self.mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
      p = norm.pdf(self.x_list, loc=self.mu, scale=sigma)
      p_dict[sigma] = p

    return p_dict

  @Plotter.property()
  def numerical_entropy_dict(self):
    # This property will not be generated until called first time
    entropy_dict = {}
    dx = self.x_list[1] - self.x_list[0]

    for sigma in self.sigma_list:
      p = self.p_dict[sigma]
      # Calculate entropy
      entropy = -np.sum(p * np.log(p + 1e-300)) * dx
      entropy_dict[sigma] = entropy

    return entropy_dict

  @Plotter.property()
  def analytical_entropy_dict(self):
    # This property will not be generated until called first time
    entropy_dict = {}

    for sigma in self.sigma_list:
      entropy = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
      entropy_dict[sigma] = entropy

    return entropy_dict


  def plot(self, x, fig: plt.Figure):
    sigma = x

    # (1) Draw distribution on the first row
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(self.x_list, self.p_dict[x])
    ax1.set_xlim([-4, 4])
    ax1.set_ylim([0, 4.2])
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('Density')

    ax1.set_title(f'Gaussian distribution $(\mu=0, \sigma={sigma:.2f})$')
    ax1.grid(True)

    # (2) Draw entropy list
    # Get entropy
    entropy_dict = (self.analytical_entropy_dict if self.get('analytical')
                    else self.numerical_entropy_dict)
    entropy_list = [entropy_dict[s] for s in self.sigma_list]

    ax2 = fig.add_subplot(2, 1, 2)

    ax2.plot(self.sigma_list, entropy_list)
    ax2.plot(sigma, self.numerical_entropy_dict[sigma], 'ro')

    ax2.set_xlabel('$\sigma$')
    ax2.set_ylabel('Entropy')
    ax2.grid(True)

    ax2.set_title(f'Entropy $H(p)=-\int p(x)\log p(x) dx={self.numerical_entropy_dict[x]:.2f}$')

    # (-1) Finalize
    fig.tight_layout()



if __name__ == '__main__':
  p = Pictor(title='Entropy', figure_size=(10, 6))

  p.add_plotter(EntropyDemo())
  p.objects = np.logspace(-1, 0, 50)[::-1]

  p.show()
