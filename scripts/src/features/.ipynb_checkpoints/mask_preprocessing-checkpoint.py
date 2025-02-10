import numpy as np
#import SimpleITK as sitk
from scipy import spatial
from PIL import Image, ImageFilter
import itk

def median_filter(inputImage):
    """NOT WORKING. Smooths the edges of mask 
    with a median filter.
    
    Parameters
    ----------
    inputImage : sitk object
        sitk binary mask.
    
    Returns
    -------
    sitk image
        sitk mask with smooth borders.
    """
    
    #tentar por slice
    arr = sitk.GetArrayFromImage(inputImage)
    image = Image.fromarray(arr, mode='L')
    new_image = image.filter(ImageFilter.ModeFilter(size=13))
    new_arr = np.asarray(new_image)
    out_sitk_image = sitk.GetImageFromArray(new_arr)
    out_sitk_image.CopyInformation(inputImage)
    return out_sitk_image

from skimage.morphology import (convex_hull_image,remove_small_objects,remove_small_holes,label)

def remove_objects(mask):
    """Removes all objects from a binary mask except 
    the largest one. (Assuming the largest object has
    the highest probability of being the correct 
    segmentation)
    
    Parameters
    ----------
    mask : sitk object
        sitk binary mask.
    
    Returns
    -------
    sitk image
        sitk mask with only the largest object.
    """
    import itk
    arr = itk.GetArrayFromImage(mask)
    arr = arr.astype(dtype=bool)
    arr, nb_objects = label(arr, connectivity=3, return_num=True)
    min_size = 0
    while nb_objects > 1:
        min_size += 100
        new_arr = remove_small_holes(remove_small_objects(arr, min_size=min_size, connectivity=3), area_threshold=min_size, connectivity=3)
        arr, nb_objects = label(new_arr, connectivity=3, return_num=True)
    arr = arr.astype(int)
    new_mask = itk.GetImageFromArray(arr.astype(np.uint8))
    new_mask.CopyInformation(mask)
    #new_mask = sitk.Cast(new_mask, sitk.sitkUInt8)
    return new_mask

def fill_holes(mask):
    """Fills small holes.
    
    Parameters
    ----------
    mask : sitk object
        sitk binary mask.
    
    Returns
    -------
    sitk image
        sitk mask without holes.
    """

    arr = sitk.GetArrayFromImage(mask)
    arr = arr.astype(dtype=bool)
    arr = label(arr, connectivity=2)
    new_mask = remove_small_holes(remove_small_objects(arr))
    hull = convex_hull_image(new_mask)
    hull = hull.astype(int)
    out_sitk_image = sitk.GetImageFromArray(hull)
    out_sitk_image.CopyInformation(mask)
    return out_sitk_image


def flood_fill_hull(sitk_image):
    """
    Flood fill an SITK image. Adapted from [1].
    [1] https://stackoverflow.com/a/46314485
    """
    # convert SITK image to np array
    image = itk.GetArrayFromImage(sitk_image)
    # extract coordinates for all positive indices
    points = np.transpose(np.where(image))
    # calculate the convex hull (a geometrical object containing all points
    # whose vertices are defined from points)
    hull = spatial.ConvexHull(points)
    # calculate the Delaunay triangulation - in short, this divides the convex
    # hull into a set of triangles whose circumcircles do not contain those of
    # other triangles
    deln = spatial.Delaunay(points[hull.vertices])
    # construct a new array whose elements are the 3d coordinates of the image
    idx = np.stack(np.indices(image.shape), axis = -1)
    # find the simplices containing these indices (idx)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    # create new image
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    # convert to SITK image and copy information
    #out_sitk_image = sitk.GetImageFromArray(out_img)
    #out_sitk_image.CopyInformation(sitk_image)
    new_mask = itk.GetImageFromArray(out_img.astype(np.uint8))
    new_mask.CopyInformation(sitk_image)
    #out_sitk_image = sitk.Cast(out_sitk_image, sitk.sitkUInt8)
    return new_mask