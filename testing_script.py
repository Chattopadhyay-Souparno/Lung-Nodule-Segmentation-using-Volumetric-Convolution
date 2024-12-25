from __future__ import print_function, division
import os
import SimpleITK as sitk
import cv2
import numpy as np
from glob import glob
from data_preprocessing.segmentation_modified import prepare_3d_training_data


def resize_image_itk(itkimage, newSpacing, resamplemethod=sitk.sitkNearestNeighbor):
   
    newSpacing = np.array(newSpacing, float)
    originSpacing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = np.array(itkimage.GetSize(), dtype=np.int)
    factor = originSpacing / newSpacing
    newSize = (originSize * factor).astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    if resamplemethod == sitk.sitkNearestNeighbor:
        itkimgResampled = sitk.Threshold(itkimgResampled, 0, 1.0, 255)
    imgResampled = sitk.GetArrayFromImage(itkimgResampled)
    return imgResampled, itkimgResampled


def load_itkfilewithtruncation(filename, upper=200, lower=-200):
   
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    sitkTruncatedImage = sitk.GetImageFromArray(srcitkimagearray)
    sitkTruncatedImage.SetSpacing(srcitkimage.GetSpacing())
    sitkTruncatedImage.SetOrigin(srcitkimage.GetOrigin())
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitkTruncatedImage, sitk.sitkFloat32))
    return itkimage


def process_original_traindata():
    trainImage_path = "preprocessed_test/image/0"
    if not os.path.exists(trainImage_path):
        os.makedirs(trainImage_path)
    
    for subsetindex in range(1):
        luna_path = "src_test/"
        luna_subset_path = luna_path + "subset_test" + "/"
        file_list = glob(luna_subset_path + "*.mhd")
        for fcount in range(len(file_list)):
            # 1. Load itk image and truncate value with upper and lower bounds
            src = load_itkfilewithtruncation(file_list[fcount], 600, -1000)
            sub_img_file = file_list[fcount][len(luna_subset_path):-4]
            
            # 2. Resample if z spacing > 1.0
            z_spacing = src.GetSpacing()[-1]
            if z_spacing > 1.0:
                _, src = resize_image_itk(src, (src.GetSpacing()[0], src.GetSpacing()[1], 1.0), resamplemethod=sitk.sitkLinear)
            
            # 3. Get resample array (image)
            srcimg = sitk.GetArrayFromImage(src)
            
            # 4. Calculate middle slices
            total_slices = srcimg.shape[0]
            middle_index = total_slices // 2
            start_index = max(0, middle_index - 20)
            end_index = min(total_slices, middle_index + 20)
            
            # 5. Save middle slices as .bmp files
            for z in range(start_index, end_index):
                slice_img = np.clip(srcimg[z], 0, 255).astype('uint8')
                cv2.imwrite(os.path.join(trainImage_path, f"{z}.bmp"), slice_img)

process_original_traindata()

def preparenoduledetectiontraindata(has_mask=True):
    height = 512
    width = 512
    number = 1
    srcpath = r"preprocessed_test/image"
    maskpath = r"preprocessed_test/mask" if has_mask else None
    trainImage = r"segmentation_test/Image"
    trainMask = r"segmentation_test/Mask" if has_mask else None
    prepare_3d_training_data(srcpath, maskpath, trainImage, trainMask, number, height, width, (16, 96, 96), 3, 20)

# Example usage:
# Call this function with has_mask=True if mask data is available, otherwise with has_mask=False
preparenoduledetectiontraindata(has_mask=False)
