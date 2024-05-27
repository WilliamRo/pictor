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
"""This module provide a wrapper for radiomic feature extraction based on
`pyradiomics` package.

Contributor: Gordon Wang & Jimmy Shen
"""
from collections import OrderedDict
from roma import Nomear, console

import numpy as np
import os
import SimpleITK as sitk



class RadiomicFeatureExtractor(Nomear):

  def __init__(self, settings={}, filters=()):
    from radiomics import featureextractor

    self.pyrad_module = featureextractor
    self.settings = settings
    self.filters = filters

    self.prompt = '[RadFeatureExtractor] >>'

  # region: Public Method

  def extract_features_from_nii(self, image_path, mask_path, mask_labels=(1,),
                                return_fmt='array&names', verbose=0):
    import SimpleITK as sitk

    # Read SimpleITK mask
    if verbose > 0: console.show_status('Reading mask ...', prompt=self.prompt)
    mask: sitk.Image = self.get_mask_from_nii(mask_path, mask_labels)

    # Read image, and align with mask
    if verbose > 0: console.show_status('Reading image ...', prompt=self.prompt)
    img = sitk.ReadImage(image_path)

    img.SetOrigin(mask.GetOrigin())
    img.SetSpacing(mask.GetSpacing())
    img.SetDirection(mask.GetDirection())

    # Extract radiomic features
    extractor = self.get_ibsi_extractor()

    if verbose > 0:
      console.show_status('Extracting features ...', prompt=self.prompt)

    result = extractor.execute(image_path, mask_path, label=1)

    if verbose > 0:
      N = len([_ for _ in result.keys() if not _.startswith('diagnostics_')])
      console.show_status(
      f'{N} features extracted.', prompt=self.prompt)

    # Return results accordingly
    assert return_fmt in ('raw', 'dict', 'array&names')
    if return_fmt == 'raw': return result

    feature_dict = OrderedDict()
    for key, value in result.items():
      if key.startswith('diagnostics_'): continue
      feature_dict[key] = float(value)

    if return_fmt == 'dict': return feature_dict

    return np.array(list(feature_dict.values())), list(feature_dict.keys())

  def get_ibsi_extractor(self):
    settings = {
      'binWidth': 25,
      # 'binWidth': 1,
      # 'additionalInfo': False,
      'normalize': True,
      'resampledPixelSpacing': [2, 2, 2],
      'Interpolator': sitk.sitkBSpline,
    }
    settings.update(self.settings)
    extractor = self.pyrad_module.RadiomicsFeatureExtractor(**settings)

    for ft in self.filters: extractor.enableImageTypeByName(ft)

    # extractor.enableImageTypeByName('LoG')
    # extractor.enableImageTypeByName('Wavelet')

    extractor.enableAllFeatures()
    return extractor

  def convert_dcm_to_nii(self, dcm_dir, save_dir=None,
                         file_name=None, verbose=False) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(dcm_dir)
    reader.SetFileNames(dcm_names)
    image = reader.Execute()

    if save_dir is not None:
      if file_name is None: file_name = os.path.basename(dcm_dir) + '.nii.gz'
      sitk.WriteImage(image, os.path.join(save_dir, file_name))
      if verbose: console.show_status(f'{file_name} saved to {save_dir}.')

    return image

  # endregion: Public Method

  # region: Utility Methods

  def get_mask_from_nii(self, mask_path, label_list, return_array=False,
                        plot=False, plot_settings={}):
    import SimpleITK as sitk

    # Read and extract mask
    mask = sitk.ReadImage(mask_path)

    mask_arr = sitk.GetArrayViewFromImage(mask)
    mask_arr = self.mask2onehot(mask_arr, label_list)

    # Plot 3D mask if required
    if plot:
      from pictor.xomics.radiomics import rad_plotters
      rad_plotters.plot_3D_mask(mask_arr, **plot_settings)

    # Return mask array or SimpleITK image
    if return_array: return mask_arr

    new_mask = sitk.GetImageFromArray(mask_arr)
    new_mask.CopyInformation(mask)
    return new_mask

  @staticmethod
  def mask2onehot(seg, labels: list):
    """Convert mask to one-hot encoding."""
    onehot = np.zeros_like(seg, dtype=bool)
    onehot[np.isin(seg, labels)] = True
    return onehot.astype(np.short)

  # endregion: Utility Methods
