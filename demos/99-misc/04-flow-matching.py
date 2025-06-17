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
r"""This script implements an intuitive flow matching demo via Pictor.

"""

from pictor import Pictor, Plotter
from torch import nn
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch



class CheckerBoardFM(Plotter):

  class MLP(nn.Module):

    class Dense(nn.Module):
      def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.ReLU()

      def forward(self, x):
        return self.activation(self.linear(x))

    # def __init__(self, n_layers=3, dim=32):
    def __init__(self, n_layers=5, dim=512):

      super().__init__()
      self.dim = dim
      self.in_projection = nn.Linear(2, dim)
      self.t_projection = nn.Linear(dim, dim)
      self.layers = nn.Sequential(*[self.Dense(dim) for _ in range(n_layers)])
      self.out_projection = nn.Linear(dim, 2)

    def get_t_embedding(self, t, max_positions=10000):
      t = t * max_positions
      half_dim = self.dim // 2
      emb = np.log(max_positions) / (half_dim - 1)
      emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
      emb = t[:, None] * emb[None, :]
      emb = torch.cat([emb.sin(), emb.cos()], dim=1)
      if self.dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1), mode='constant', value=0.0)
      return emb

    def forward(self, x, t):
      t = self.get_t_embedding(t)
      t = self.t_projection(t)
      x = self.in_projection(x)
      assert t.shape == x.shape
      x = x + t
      x = self.layers(x)
      return self.out_projection(x)

  N = 1000
  length = 4
  v_max = 4.0

  def __init__(self, pictor=None):
    super().__init__(self.plot, pictor)
    self.traj = None
    _ = self.gaussian_data

    self.new_settable_attr('dataset', False, bool,
                           'Option to show dataset')


  @Plotter.property()
  def sample_points(self):
    checkerboard = np.indices((self.length, self.length)).sum(axis=0) % 2

    sample_points = []
    while len(sample_points) < self.N:
      x_sample = np.random.uniform(-self.v_max, self.v_max)
      y_sample = np.random.uniform(-self.v_max, self.v_max)

      i = int((x_sample + self.v_max) / (2 * self.v_max) * self.length)
      j = int((y_sample + self.v_max) / (2 * self.v_max) * self.length)

      if checkerboard[j, i] == 1: sample_points.append([x_sample, y_sample])

    return np.array(sample_points)


  @Plotter.property()
  def gaussian_data(self):
    # data = np.random.randn(self.N, 2)
    data = torch.randn(self.N, 2, dtype=torch.float32)
    self.traj = [data]
    return data


  @Plotter.property()
  def model(self): return self.MLP()


  def train(self, steps: int = 10_000, lr: float = 1e-4, batch_size: int = 64):
    optim = torch.optim.AdamW(self.model.parameters(), lr=lr)

    data = torch.Tensor(self.sample_points)
    pbar = tqdm(range(steps), desc='Training')
    losses = []
    for _ in pbar:
      x1 = data[torch.randint(data.size(0), (batch_size,))]
      x0 = torch.randn_like(x1)
      target = x1 - x0
      t = torch.rand(batch_size, dtype=torch.float32)
      xt = (1 - t[:, None]) * x0 + t[:, None] * x1
      pred = self.model(xt, t)
      loss = ((target - pred) ** 2).mean()
      loss.backward()
      optim.step()
      optim.zero_grad()

      pbar.set_postfix(loss=loss.item())
      losses.append(loss.item())

    print(f'Training finished after {steps} steps.')


  def flow(self, steps: int = 50):
    xt = self.gaussian_data
    self.traj = [xt]
    pbar = tqdm(torch.linspace(0, 1, steps)[1:], desc='Creating Flow')
    for t in pbar:
      pred = self.model(xt, t.expand(xt.size(0)))
      xt = xt + (1 / steps) * pred
      self.traj.append(xt.detach().numpy())

    print(f'Flow creation finished ({steps} steps).')
    self.pictor.objects = list(range(len(self.traj)))
    self.refresh()


  def plot(self, x, ax: plt.Axes):
    configs = {'alpha': 0.5, 's': 10}

    if self.get('dataset'):
      ax.scatter(self.sample_points[:, 0], self.sample_points[:, 1], **configs)
      ax.set_title('Data Set')
    else:
      data = self.traj[x]
      if x > 0: t = x / (len(self.traj) - 1)
      else: t = 0
      ax.scatter(data[:, 0], data[:, 1], **configs)
      ax.set_title(f'x (t={t:.2f})')

    ax.grid(True)
    ax.set_xlim([-self.v_max, self.v_max])
    ax.set_ylim([-self.v_max, self.v_max])


  def show_time_embedding(self):
    mlp = self.model
    max_positions = 1001
    t = np.linspace(0, 1, max_positions)
    emb = mlp.get_t_embedding(torch.tensor(t, dtype=torch.float32))
    emb = emb.numpy()

    # Show in matplotlib using image show and color bar
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(emb.T, aspect='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, orientation='vertical')

    # Set x_ticks from 0 to 1
    ax.set_xticks(np.linspace(0, max_positions - 1, 5))
    # Modify ticks labels
    ax.set_xticklabels([f'{i / 4}' for i in range(5)])
    ax.set_xlabel('Time')
    ax.set_ylabel('Dimension')
    ax.set_title('Time Embedding Visualization')
    fig.subplots_adjust()
    plt.tight_layout()
    plt.show()
  ste = show_time_embedding


  def register_shortcuts(self):
    super().register_shortcuts()
    self.register_a_shortcut('space', lambda: self.flip('dataset'),
                             'Toggle `dataset`')



if __name__ == '__main__':
  p = Pictor()

  cb = CheckerBoardFM()

  p.add_plotter(cb)

  p.objects = [0]

  p.show()

