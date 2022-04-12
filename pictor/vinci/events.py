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
# =-===========================================================================-
from pictor.vinci.board import Board
from typing import Optional

import inspect
import matplotlib.pyplot as plt


class StateMachine(object):

  def __init__(self, buffer_len: int = 100):
    # Library is a list of callable functions for handling events
    self.library = {}
    # Other attributes
    self._buffer_len = buffer_len
    self.key_buffer = []
    self.board: Optional[Board] = None

  # region: Public Methods

  def bind_key_press_event(self, board: Optional[Board] = None):
    # Get canvas to bind
    if board is None:
      canvas = plt.gcf().canvas
    else:
      canvas = board.canvas
      self.board = board

    # Disconnect default commands
    canvas.mpl_disconnect(canvas.manager.key_press_handler_id)

    # Bind event to canvas
    canvas.mpl_connect(
      'key_press_event', lambda e: self._on_key_press(e))

  def register_key_event(self, key, method):
    if key in self.library:
      raise KeyError('!! Key `{}` has already been registered')
    assert callable(method)
    self.library[key] = method

  # endregion: Public Methods

  # region: Private Methods

  def _on_key_press(self, event):
    key = event.key
    # Put key into buffer
    if key in self.library:
      method = self.library[key]
      # Call method with proper arguments
      kwargs = self._get_kwargs_for_event(method)
      method(**kwargs)
    elif key in ('escape_', 'q'):
      plt.close()
    elif key == 'ctrl+enter':
      self.board.window.state('zoomed')
    elif key not in ('control', 'alt', 'shift'):
      # Ignore modifiers
      print('>> key "{}" pressed'.format(key))

    # Things should be done inside each branch, such as refresh

  def _get_kwargs_for_event(self, method):
    # TODO: raise error if default value is not provided
    assert callable(method)
    # Get method signature
    sig = inspect.signature(method).parameters
    # Get key-word arguments for method according to its signature
    kwargs = {}
    for kw in sig.keys():
      if kw in ('state_machine', 'sm'):
        kwargs[kw] = self
      elif kw in ('board', 'da_vinci', 'vinci'):
        kwargs[kw] = self.board
      # else: raise KeyError('!! Illegal argument name `{}`'.format(kw))
    return kwargs

  # endregion: Private Methods





