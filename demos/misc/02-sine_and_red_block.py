from pictor import Pictor

import matplotlib.pyplot as plt
import numpy as np



t = np.linspace(0, 5, 200)

func_1 = lambda x: np.sin(1.0 * x)
func_2 = lambda x: np.sin(2.0 * x)

def plotter(x, ax: plt.Axes, func):
  ax.plot(t, func(t), 'b-')
  ax.plot(x, func(x), 'rs')

  ax.set_xlim(t[0], t[-1])
  ax.grid(True)


p = Pictor('Sine and red block', figure_size=(8, 4))
p.objects = t
p.add_plotter(lambda x, ax: plotter(x, ax, func_1))
p.add_plotter(lambda x, ax: plotter(x, ax, func_2))
p.show()