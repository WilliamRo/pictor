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
# ====-==================================================================-======
from collections import OrderedDict

import numpy as np

from roma import Nomear
from roma import check_type



class ParticleSystem(Nomear):

  CONSTANTS = {}
  DEFAULT_TRACK = 'main'

  def __init__(self, num_particles):
    self.num_particles = check_type(num_particles, int)

  # region: Properties

  @Nomear.property(local=True)
  def timelines(self): return OrderedDict()

  @Nomear.property(local=True)
  def variable_shapes(self): return OrderedDict()

  # endregion: Properties

  # region: Public Methods

  def register_var(self, k, shape, init_value=None, dtype=float,
                   is_global_var=False,):
    """Initialize a variable.

    :param k: variable key
    :param shape: variable shape
    :param init_value: initial values. Will be set to zeros if not provided.
    :param dtype: data type
    :param is_global_var: whether this var is a global var
    """
    # Register shape
    if k in self.variable_shapes:
      raise KeyError(f'!! Variable `{k}` already exists')

    if shape is None: self.variable_shapes[k] = None
    else:
      if isinstance(shape, int): shape = [shape]
      if not is_global_var: shape = [self.num_particles] + list(shape)
      self.variable_shapes[k] = tuple(shape)

    # Set init_value to 0 if not provided
    if init_value is None: init_value = np.zeros(shape, dtype=dtype)
    else: init_value: np.ndarray = np.array(init_value, dtype=dtype)

    # Initialize variable k
    self.set_variables(0, **{k: init_value})

  def register_constants(self, **constants):
    for k, v in constants.items(): self.CONSTANTS[k] = v

  def get_values(self, t, *keys, track=DEFAULT_TRACK):
    """Get a list of variables in time step t"""
    value_dict = self._get_variable_dict(t, track)
    value_dict.update(self.CONSTANTS)

    return [value_dict.get(k, None) for k in keys]

  def set_variables(self, t, track=DEFAULT_TRACK, **kwargs):
    """Set variables in time step t."""

    var_dict = self._get_variable_dict(t, track)
    for k, v in kwargs.items():
      # Check key
      if k in self.CONSTANTS: raise KeyError(f'!! `{k}` is a system constant.')
      # Check value
      if k not in self.variable_shapes:
        raise KeyError(f'!! `{k}` has not been registered yet')
      shape = self.variable_shapes[k]

      # If valid shape is provided, make sure v.shape == valid shape
      if shape is not None:
        if not isinstance(v, np.ndarray):
          raise TypeError(f'!! `{k}` should be an np.ndarray')
        if shape != v.shape: raise AssertionError(
          f'!! `{k}`.shape should be {shape} instead of {v.shape}')

      # Set variable
      var_dict[k] = v

  # endregion: Public Methods

  # region: Private Methods

  def _get_variable_dict(self, t, track=DEFAULT_TRACK) -> dict:
    # Select timeline
    if track not in self.timelines: self.timelines[track] = OrderedDict()
    timeline = self.timelines[track]
    # Return variables on time step t
    t = float(t)
    if t not in timeline: timeline[t] = OrderedDict()
    return timeline[t]

  def _sanity_check(self): pass

  def _move_forward(self, dt, track=DEFAULT_TRACK):
    """Calculate system states on time step t + dt,
    where t is the latest time step of specified track"""
    # Get package
    last_t = list(self.timelines[track].keys())[-1]
    pkg = {**self._get_variable_dict(last_t, track)}
    pkg.update(self.CONSTANTS)

    # Run simulation
    new_states: dict = self.simulate(pkg, dt)

    # Set variables
    self.set_variables(last_t + dt, track, **new_states)

  # endregion: Private Methods

  # region: Abstract Methods

  @staticmethod
  def simulate(pkg: dict, dt: float) -> dict:
    """Simulate next time step based on current step"""
    raise NotImplementedError

  # endregion: Abstract Methods



if __name__ == '__main__':
  pass
