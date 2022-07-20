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
# =====-===========================================================-============
from pictor import Pictor
from pictor.zebra.io.inflow import Inflow

from typing import Optional, List



class Zebra(Pictor):

  def __init__(self, title='Zebra'):
    # Call parent's constructor
    super(Zebra, self).__init__(title)

    # Special Attributes
    self.inflow: Optional[Inflow] = None

  # region: Public Methods

  def set_inflow(self, inflow: Inflow):
    assert isinstance(inflow, Inflow)
    self.inflow = inflow

  # endregion: Public Methods

  # region: Shortcuts

  def _register_default_key_events(self):
    super(Zebra, self)._register_default_key_events()

    self.shortcuts.register_key_event(' ', None)

  # endregion: Shortcuts

