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
# ===-=====================================================================-====
"""This class provides filter functions for subclasses of Pictor"""
from roma import Nomear

import inspect



class Database(Nomear):

  @Nomear.property()
  def all_objects(self): return self.objects


  @Nomear.property()
  def all_labels(self): return self.labels


  def filter_by_func(self, func=lambda _: True):
    """Filter objects by given function.
    func should be callable function with 1 or 2 input parameters, and
    should return a boolean value. If it has 2 inputs, the name of second input
    parameter should be `label`
    """
    from pictor.pictor import Pictor
    assert isinstance(self, Pictor) and isinstance(self, Database)

    all_objects = self.all_objects
    all_labels = self.all_labels

    # Inspect func, and find indices accordingly
    para_names = list(inspect.signature(func).parameters.values())
    if len(para_names) == 1:
      indices = [i for i, obj in enumerate(all_objects) if func(obj)]
    else:
      if not (len(para_names) == 2 and para_names[1] == 'label'):
        raise AssertionError('!! Illegal filter function {}({})'.format(
          func.__name__, ', '.join(para_names)))
      indices = [
        i for i, (obj, label) in enumerate(zip(all_objects, all_labels))
        if func(obj, label)]

    # Set filtered objects
    self.objects = [all_objects[i] for i in indices]

    # Filter labels if necessary
    if all_labels is not None:
      self.labels = [all_labels[i] for i in indices]

    # Set cursor to 0 and refresh
    self.set_cursor(self.Keys.OBJECTS, cursor=0)
    self.refresh()


  def filter_by_label(self, contain: str = ''):
    """Filter objects by label"""
    if self.labels is None: raise AssertionError('!! Objects have no label')
    self.filter_by_func(lambda _, label: contain in label)



