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
# ====-======================================================================-==
"""This module provides file-exploring functions for Pictor APPs"""




class DialogUtilities(object):

  @staticmethod
  def load_file_dialog(title):
    from tkinter.filedialog import askopenfilename
    return askopenfilename(title=title)

  @staticmethod
  def select_folder_dialog(title):
    from tkinter.filedialog import askdirectory
    return askdirectory(title=title)
