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
# ===-=======================================================================-==
import numpy as np



def plot_3D_mask(mask, step_size=5, fig_size=(6, 6), alpha=0.5,
                 face_color=(0.8, 0, 0)):
  """Ref: https://www.youtube.com/watch?v=WBJH_B-CYLY"""
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  from skimage.measure import marching_cubes

  import matplotlib.pyplot as plt

  # Extract the mesh
  verts, faces, _, _ = marching_cubes(mask, step_size=step_size)
  mesh = Poly3DCollection(verts[faces], alpha=alpha)
  mesh.set_facecolor(face_color)

  # Plot the mesh
  fig = plt.figure(figsize=fig_size)
  ax = fig.add_subplot(111, projection='3d')

  ax.add_collection3d(mesh)
  ax.set_xlim(0, mask.shape[0])
  ax.set_ylim(0, mask.shape[1])
  ax.set_zlim(0, mask.shape[2])

  plt.tight_layout()
  plt.show()
