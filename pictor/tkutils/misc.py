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
# ==-=======================================================================-===
def centerize_window(window):
  h, w = window.winfo_height(), window.winfo_width()
  H, W = window.winfo_screenheight(), window.winfo_screenwidth()
  x, y = (W - w) // 2, (H - h) // 2
  window.geometry("+{}+{}".format(x, y))


def show_elegantly(window):
  window.focus_force()
  window.after(1, lambda: centerize_window(window))
  window.mainloop()

