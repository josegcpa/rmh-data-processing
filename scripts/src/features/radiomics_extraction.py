import os
import re
import sys
import six
import itk
import math
import random
import logging
import warnings
import radiomics
import numpy as np
import pandas as pd
import SimpleITK as sitk
import src.features.mask_preprocessing as mpp
from radiomics import featureextractor
from alive_progress import alive_bar

def checkSpacing(images_files):
	"""Checks the images' voxel spacing and saves all 
	the x-, y- and z-spacings into three pandas series.

	Parameters
	----------
	images_files : lst
		List with the relative location of each image.
	directory : str
		Root directory where relative paths will be appended.

	Returns
	-------
	pandas series, pandas series, pandas series
		the first has the x-spacing values for all images.
		the second has the y-spacing values for all images.
		the third has the z-spacing values for all images.
	"""
	
	x=[]
	y=[]
	z=[]
	with alive_bar(len(images_files), title='Checking image spacing', force_tty=True) as bar:
		for image_path in images_files:
			inputImage = sitk.ReadImage(image_path, sitk.sitkFloat32)
			x.append(inputImage.GetSpacing()[0])
			y.append(inputImage.GetSpacing()[1])
			z.append(inputImage.GetSpacing()[2])
			bar()
	return pd.Series(x), pd.Series(y), pd.Series(z)

def bias_field_correction(inputImage, inputMask):
    """Uses the N4-BiasFieldCorrection algorithm to correct
    the input image, and returns the corrected image.
    
    Parameters
    ----------
    inputImage : sitk object
        sitk T2 image.
    inputMask : sitk object
        sitk binary mask.
        
    Returns
    -------
    sitk image
        Corrected T2 image.
    """
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(inputImage, inputMask)
    return corrected_image

def getRange(images_files, masks_files, bfcorrection = False, coreg = False, coreg_images = False):
    """Uses PyRadiomics to extract the intensity range for 
    all the image filters of a series of images.

    Parameters
    ----------
    images_files : lst
        List with the relative location of each image.
    masks_files : lst
        List with the relative location of each mask.
    directory : str
        Root directory where relative paths will be appended.
    bfcorrection : bool, optional
        Whether to do bias field correction on image before 
        extracting the intensity range. Should be set to True 
        for T2W images.

    Returns
    -------
    pandas DataFrame
        Row index is the image's relative path. One column 
        per image filter. The cell value is the calculated
        range.
    """
    
    features = {}
    with alive_bar(len(images_files), title='Calculating intensity ranges', force_tty=True) as bar:
        for idx in range(len(images_files)):
            
            #LOAD IMAGES
            try:
                fixed_image = itk.imread(images_files[idx], itk.F)
                moving_mask = itk.imread(masks_files[idx], itk.F)
                itk.imwrite(fixed_image, 'test_image.nii')
            except:
                print('error in load', images_files[idx])
                bar()
                continue
            
            if coreg:
                try:
                    # CO-REGISTRATION
                    moving_image = itk.imread(coreg_images[idx], itk.F)
                    parameter_object = itk.ParameterObject.New()
                    parameter_object.AddParameterFile('../data/params/coreg_params.txt')
                    result_image, result_transform_parameters = itk.elastix_registration_method(
                        fixed_image, moving_image, parameter_object=parameter_object)
                    result_transform_parameters.SetParameter('FinalBSplineInterpolationOrder','0')
                    transform_mask = itk.transformix_filter(moving_mask, result_transform_parameters)
                    itk.imwrite(fixed_image, 'test_image.nii')
                except:
                    print('error in coreg', images_files[idx])
                    bar()
                    continue
                
                try:
                    # POST-PROCESSING
                    processed_mask = mpp.flood_fill_hull(mpp.remove_objects(transform_mask))
                    itk.imwrite(processed_mask, 'test_mask_processed.nii')
                except:
                    print('error in post-processing', images_files[idx])
                    bar()
                    continue
                
            else:
                # POST-PROCESSING
                try:
                    processed_mask = mpp.flood_fill_hull(mpp.remove_objects(moving_mask))
                    itk.imwrite(processed_mask, 'test_mask_processed.nii')
                except:
                    print('error in post-processing', images_files[idx])
                    bar()
                    continue
            
            # BIAS-FIELD CORRECTION
            if bfcorrection:
                try:
                    temp_image = sitk.ReadImage("test_image.nii")
                    temp_mask = sitk.ReadImage("test_mask_processed.nii")
                    inputImage = bias_field_correction(temp_image, temp_mask)
                    sitk.WriteImage(inputImage, 'test_image.nii')
                except:
                    print('error in bfc', images_files[idx])
                    bar()
                    continue
            
            # EXTRACTION
            extr = featureextractor.RadiomicsFeatureExtractor(normalize = True, force2D = True, 
                                                              normalizeScale = 100, voxelArrayShift = 300)
            extr.enableAllImageTypes()
            extr.disableAllFeatures()
            extr.enableFeaturesByName(firstorder=['Range'])
            try:
                res = extr.execute('test_image.nii', 'test_mask_processed.nii')
            except:
                print('error in extractor', images_files[idx])
                bar()
                continue
            features[images_files[idx]] = res
            bar()
    df = pd.DataFrame(features, columns = features.keys()).transpose()

    for col in df.columns:
        if re.search('diagnostics', col):
            del df[col]
    
    return df

def closest_value(input_list, input_value):
	"""Finds the value in a list that's closest to a given
	number.

	Parameters
	----------
	input_list : lst
		List with floats.
	input_value : int
		Number to which we want to find the closest value.
	
	Returns
	-------
	int
		Index of closest value.
	"""
	
	arr = np.asarray(input_list)
	i = (np.abs(arr - input_value)).argmin()
	return i

def checkBinWidth(ranges):
	"""Calculates the number of bins for a series of 
	binwidths for all images. Selects the binwidth that
	results in the average number of bins that is closest
	to 80, for each image filter.

	Parameters
	----------
	ranges : pandas DataFrame
		Dataframe with the intensity range of each image 
		with each filter.

	Returns
	-------
	dict
		Dictionary that connects each image filter (key)
		with the selected binwidth (value).
	"""

	w = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 1, 3, 4, 5, 8, 12, 16, 20, 25, 30, 50, 70, 100]
	filters = ranges.columns
	res={}
	with alive_bar(len(filters), title='Calculating binwidths', force_tty=True) as bar:
		for f in filters:
			r = ranges[[f]]
			b = pd.DataFrame()
			for i in w:
				b[str(i)] = pd.to_numeric(r[f]/i)
			desc = b.describe().transpose()
			i = closest_value(desc['mean'], 80)
			res[f] = w[i]
			bar()
	return res

def load_images(image_path, mask_path):
	"""Checks that the image's and mask's paths exist, 
	then loads them as sitk images and returns them.

	Parameters
	----------
	image_file : str
		Relative location of the image.
	mask_file : str
		Relative location of the mask.
	directory : str
		Root directory where relative paths will be appended.
	
	Returns
	-------
	sitk image, sitk image
		Loaded image and respective mask.
	"""

	if os.path.exists(image_path) and os.path.exists(mask_path):
		inputImage = sitk.ReadImage(image_path, sitk.sitkFloat32)
		inputMask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
		#inputMask.SetOrigin(inputImage.GetOrigin())
		return inputImage, inputMask
	else:
		new_path = image_path.split('/')[0:-2]
		new_path.append('image_DWI.nii.gz')
		new_path = os.path.join(*new_path)
		if os.path.exists(new_path):
			inputImage = sitk.ReadImage(new_path, sitk.sitkFloat32)
			inputMask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
			#inputMask.SetOrigin(inputImage.GetOrigin())
			return inputImage, inputMask
		else:
			return None, None

def extractor(images_files, masks_files, settings_path = None, bfcorrection = False, coreg = False, coreg_images_moving = None, coreg_images_fixed = None):
    """Uses PyRadiomics to extract the radiomic features of a 
    series of images.

    Parameters
    ----------
    images_files : lst
        List with the relative location of each image.
    masks_files : lst
        List with the relative location of each mask.
    directory : str
        Root directory where relative paths will be appended.
    settings_path : str, optional
        Path to settings file.
    bfcorrection : bool, optional
    Whether to do bias field correction on image before 
        extracting the intensity range. Should be set to True 
        for T2W images.

    Returns
    -------
    pandas DataFrame
        Row index is the image's patient id. One column per
        radiomic feature. The cell is the calculated value.
    """

    radiomics.setVerbosity(60)
    features = {}
    with alive_bar(len(images_files), title='Calculating radiomic features', force_tty=True) as bar:
        
        i=0
        
        for idx in range(len(images_files)):
        
            #LOAD IMAGES
            try:
                fixed_image = itk.imread(images_files[idx], itk.F)
                moving_mask = itk.imread(masks_files[idx], itk.F)
                itk.imwrite(fixed_image, 'test_image.nii')
                #print('fixed image:', images_files[idx])
                #print('moving_mask:', masks_files[idx])
            except:
                print('error in load', images_files[idx])
                bar()
                continue
            
            if coreg:
                try:
                    # CO-REGISTRATION
                    moving_image = itk.imread(coreg_images_moving[idx], itk.F)
                    #print('moving image:', coreg_images_moving[idx])
                    if coreg_images_fixed is not None:
                        fixed_image = itk.imread(coreg_images_fixed[idx], itk.F)
                        #print('new fixed image:', coreg_images_fixed[idx])
                    parameter_object = itk.ParameterObject.New()
                    parameter_object.AddParameterFile('../data/params/coreg_params.txt')
                    result_image, result_transform_parameters = itk.elastix_registration_method(
                        fixed_image, moving_image, parameter_object=parameter_object)
                    result_transform_parameters.SetParameter('FinalBSplineInterpolationOrder','0')
                    transform_mask = itk.transformix_filter(moving_mask, result_transform_parameters)
                except:
                    
                    try:
                        # CENTER-CROP
                        #print('Trying center crop')
                        if coreg_images_fixed is not None:
                            img = sitk.ReadImage(coreg_images_fixed[idx])
                            #print('image to crop:', coreg_images_fixed[idx])
                        else:
                            img = sitk.ReadImage(images_files[idx])
                            #print('image to crop:', images_files[idx])
                        size = img.GetSize()
                        spacing = img.GetSpacing()
                        crop_size = 240 / spacing[0]
                        center_x = size[0] // 2
                        center_y = size[1] // 2
                        left = int(max(center_x - crop_size // 2, 0))
                        right = int(min(center_x + crop_size // 2, size[0]))
                        top = int(max(center_y - crop_size // 2, 0))
                        bottom = int(min(center_y + crop_size // 2, size[1]))
                        cropped_image = img[top:bottom, left:right]
                        sitk.WriteImage(cropped_image, 'test_image_cropped.nii')

                        # CO-REGISTRATION
                        fixed_image = itk.imread('test_image_cropped.nii', itk.F)
                        moving_image = itk.imread(coreg_images_moving[idx], itk.F)
                        #print('moving image:', coreg_images_moving[idx])
                        parameter_object = itk.ParameterObject.New()
                        parameter_object.AddParameterFile('../data/params/coreg_params.txt')
                        result_image, result_transform_parameters = itk.elastix_registration_method(
                            fixed_image, moving_image, parameter_object=parameter_object)
                        result_transform_parameters.SetParameter('FinalBSplineInterpolationOrder','0')
                        transform_mask = itk.transformix_filter(moving_mask, result_transform_parameters)

                    except:
                        print('error in coreg', images_files[idx])
                        bar()
                        continue
                
                try:
                    # POST-PROCESSING
                    processed_mask = mpp.flood_fill_hull(mpp.remove_objects(transform_mask))
                    itk.imwrite(processed_mask, 'test_mask_processed.nii')
                except:
                    print('error in post-processing', images_files[idx])
                    bar()
                    continue
                
            else:
                # POST-PROCESSING
                try:
                    processed_mask = mpp.flood_fill_hull(mpp.remove_objects(moving_mask))
                    itk.imwrite(processed_mask, 'test_mask_processed.nii')
                except:
                    print('error in post-processing', images_files[idx])
                    bar()
                    continue
            
            # BIAS-FIELD CORRECTION
            if bfcorrection:
                try:
                    temp_image = sitk.ReadImage("test_image.nii")
                    temp_mask = sitk.ReadImage("test_mask_processed.nii")
                    inputImage = bias_field_correction(temp_image, temp_mask)
                    sitk.WriteImage(inputImage, 'test_image.nii')
                except:
                    print('error in bfc', images_files[idx])
                    bar()
                    continue
            
            # EXTRACTION
            extr = featureextractor.RadiomicsFeatureExtractor(settings_path)
            try:
                res = extr.execute('test_image.nii', 'test_mask_processed.nii')
            except:
                print('error in extractor', images_files[idx])
                df = pd.DataFrame(features, columns = features.keys()).transpose()
                for col in df.columns:
                    if re.search('diagnostics', col):
                        del df[col]
                df.to_csv(os.path.join('radiomic_features_'+str(random.randint(0, 100))+'.csv'))
                bar()
                continue
                
            features[images_files[idx].split('/')[6]] = res
            
            # SAVE EVERY 200 INSTANCES JUST IN CASE
            if i%200 == 0:
                df = pd.DataFrame(features, columns = features.keys()).transpose()
                for col in df.columns:
                    if re.search('diagnostics', col):
                        del df[col]
                df.to_csv(os.path.join('radiomic_features_'+str(i)+'.csv'))
            i+=1
            
            bar()
    
    df = pd.DataFrame(features, columns = features.keys()).transpose()
    for col in df.columns:
        if re.search('diagnostics', col):
            del df[col]
    
    return df