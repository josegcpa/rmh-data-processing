import numpy as np
import SimpleITK as sitk
from pathlib import Path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--idx", type=int, default=1)
    args = parser.parse_args()

    for npz_path in Path(args.input_path).rglob("*npz"):
        npz_path = str(npz_path)
        npz_data = np.load(npz_path)
        image = npz_data["probabilities"][args.idx]
        sitk_original = sitk.ReadImage(npz_path.replace("npz", "nii.gz"))
        image = sitk.GetImageFromArray(image)
        image.CopyInformation(sitk_original)
        sitk.WriteImage(image, npz_path.replace(".npz", "_proba.nii.gz"))
