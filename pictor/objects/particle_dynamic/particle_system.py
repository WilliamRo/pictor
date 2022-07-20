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
from roma import Nomear



class ParticleSystem(Nomear):

  def __init__(self):
    self.particle_state = OrderedDict()
    self.constants = OrderedDict()
    self.system_conditions = OrderedDict()

  # region: Properties

  # endregion: Properties

  # region: Public Methods

  # endregion: Public Methods

  # region: Abstract Methods

  def state_transition(self, state):
    state['v'] = state['v'] + state['f'] * self.constants['alpha']
    return state

  def calculate_step(self):
    self.particle_state = self.state_transition(self.particle_state)

  # endregion: Abstract Methods



if __name__ == '__main__':
  pass
