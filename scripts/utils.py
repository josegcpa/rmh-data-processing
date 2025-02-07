import pydicom
from pydicom import dcmread
from pydicom.dataset import Dataset

diffusion_bvalue = (0x0018, 0x9087)
diffusion_bvalue_ge = (0x0043, 0x1039)
diffusion_bvalue_siemens = (0x0019, 0x100C)


def process_bvalue_ge(v: bytes | float | int) -> float | int:
    """
    Process GE scanner b-value from DICOM header.

    Args:
        v (bytes | float | int): b-value.
        dicom_file (None): only for compatibility purposes.

    Returns:
        float | int: processed b-value.
    """
    if isinstance(v, bytes):
        v = v.decode().split("\\")[0]
    elif isinstance(v, pydicom.multival.MultiValue):
        v = v[0]
    if len(str(v)) > 5:
        v = str(v)[-4:].lstrip("0")
    return v


def process_bvalue(v: bytes | float | int, dicom_file: Dataset) -> float | int:
    """
    Process b-value from DICOM header.

    Args:
        v (bytes | float | int): b-value.
        dicom_file (Dataset): DICOM file.

    Returns:
        float | int: processed b-value.
    """
    if isinstance(v, bytes):
        v = int.from_bytes(v, byteorder="big")
        if v > 5000:
            v = dicom_file[diffusion_bvalue].value[0]
    return v


def retrieve_bvalue(dicom_file_path: str) -> float | int:
    """
    Retrieve b-value from DICOM file.

    Args:
        dicom_file_path (str): path to DICOM file.

    Returns:
        float | int: b-value.
    """
    dicom_file = dcmread(dicom_file_path, stop_before_pixels=True)
    if diffusion_bvalue in dicom_file:
        return process_bvalue(dicom_file[diffusion_bvalue].value, dicom_file)
    elif diffusion_bvalue_ge in dicom_file:
        return process_bvalue_ge(
            dicom_file[diffusion_bvalue_ge].value, dicom_file
        )
    elif diffusion_bvalue_siemens in dicom_file:
        return process_bvalue(
            dicom_file[diffusion_bvalue_siemens].value, dicom_file
        )
    return 0.0
