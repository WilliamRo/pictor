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
# ==-==========================================================================-
from pictor.plugins import Timer
from roma import check_type
from roma.spqr.threading import XNode



class Inflow(Timer, XNode):
  """Data inflow."""

  def __init__(self, max_len=20):
    self.buffer: list = []
    self.max_len = check_type(max_len, int)


  def append_to_buffer(self, data):
    self.buffer.append(data)
    if len(self.buffer) > self.max_len: self.buffer.pop(0)


  def fetch(self, async_=False):
    if async_:
      self.execute_async(self.fetch)
      return

    self._init()
    while not self.should_terminate:
      self._loop()
      self._tic()
    self._finalize()


  def stop_fetch(self): self.terminate()


  def _init(self): pass


  def _loop(self): raise NotImplemented


  def _finalize(self): pass
