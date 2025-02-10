import itk
import radiomics
import numpy as np
import pandas as pd
import SimpleITK as sitk

from radiomics import featureextractor


def load_image(path):
    sitk_image = sitk.ReadImage(path, sitk.sitkFloat32)
    itk_image = itk.imread(path, itk.F)
    return sitk_image, itk_image


def load_mask(path):
    sitk_image = sitk.ReadImage(path, sitk.sitkUInt8)
    itk_image = itk.imread(path, itk.F)
    return sitk_image, itk_image


def coregister(_fixed_image, _moving_image, _moving_mask):
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile("params/coreg_params.txt")
    result_image, result_transform_parameters = (
        itk.elastix_registration_method(
            _fixed_image, _moving_image, parameter_object=parameter_object
        )
    )
    result_transform_parameters.SetParameter(
        "FinalBSplineInterpolationOrder", "0"
    )
    transform_mask = itk.transformix_filter(
        _moving_mask, result_transform_parameters
    )
    return transform_mask


def radiomics_extraction(_t2w, _t2_mask, _dwi, _dwi_mask, _adc, _adc_mask):
    # BIAS FIELD CORRECTION OF T2W
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    t2w = corrector.Execute(_t2w, _t2_mask)

    # EXTRACTION
    radiomics.setVerbosity(60)

    extr = featureextractor.RadiomicsFeatureExtractor(
        "params/T2W_extr_params.yaml"
    )
    res_t2w = extr.execute(t2w, _t2_mask)
    res_t2w = del_diagnostics(res_t2w)
    res_t2w = pd.Series(res_t2w).to_frame().T
    res_t2w.columns = ["T2W_" + x for x in res_t2w.columns]

    extr = featureextractor.RadiomicsFeatureExtractor(
        "params/DWI_extr_params.yaml"
    )
    res_dwi = extr.execute(_dwi, _dwi_mask)
    res_dwi = del_diagnostics(res_dwi)
    res_dwi = pd.Series(res_dwi).to_frame().T
    res_dwi.columns = ["DWI_" + x for x in res_dwi.columns]

    extr = featureextractor.RadiomicsFeatureExtractor(
        "params/ADC_extr_params.yaml"
    )
    res_adc = extr.execute(_adc, _adc_mask)
    res_adc = del_diagnostics(res_adc)
    res_adc = pd.Series(res_adc).to_frame().T
    res_adc.columns = ["ADC_" + x for x in res_adc.columns]

    final = pd.concat([res_t2w, res_dwi, res_adc], axis=1)
    return final


def del_diagnostics(dic):
    delete = [k for k in dic if "diagnostics" in k]
    for k in delete:
        del dic[k]
    return dic


def dicom_orientation_to_sitk_direction(
    orientation: list[float],
) -> np.ndarray:
    """Converts the DICOM orientation to SITK orientation. Based on the
    nibabel code that does the same. DICOM uses a more economic encoding
    as one only needs to specify two of the three cosine directions as they
    are all orthogonal. SITK does the more verbose job of specifying all three
    components of the orientation.
    Args:
        orientation (Sequence[float]): DICOM orientation.
    Returns:
        np.ndarray: SITK (flattened) orientation.
    """
    # based on nibabel documentation
    orientation = np.array(orientation).reshape(2, 3).T
    R = np.eye(3)
    R[:, :2] = np.fliplr(orientation)
    R[:, 2] = np.cross(orientation[:, 1], orientation[:, 0])
    R_sitk = np.stack([R[:, 1], R[:, 0], -R[:, 2]], 1)
    return R_sitk.flatten().tolist()


def itk_to_sitk(itk_image):
    sitk_image = sitk.GetImageFromArray(
        itk.GetArrayFromImage(itk_image),
        isVector=itk_image.GetNumberOfComponentsPerPixel() > 1,
    )
    sitk_image.SetOrigin(tuple(itk_image.GetOrigin()))
    sitk_image.SetSpacing(tuple(itk_image.GetSpacing()))
    sitk_image.SetDirection(
        itk.GetArrayFromMatrix(itk_image.GetDirection()).flatten()
    )
    return sitk_image
