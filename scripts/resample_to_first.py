import SimpleITK as sitk
from pathlib import Path
from glob import glob

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--series", type=str, required=True, nargs="+")

    args = parser.parse_args()

    all_image_identifiers = args.series
    if len(all_image_identifiers) == 1:
        print(
            "Skipping resampling since only one image identifier is provided"
        )
        exit(0)
    all_image_identifiers.sort()
    all_images = {}
    for image_path in Path(args.input_dir).glob("*.nii.gz"):
        image_data = image_path.stem.split("_")
        root = "_".join(image_data[:-1])
        image_identifier = image_data[-1][:4]
        if root not in all_images:
            all_images[root] = {}
        all_images[root][image_identifier] = image_path

    if len(all_image_identifiers) > 1:
        for root in all_images:
            target_image = sitk.ReadImage(
                all_images[root][all_image_identifiers[0]]
            )
            for image_identifier in all_image_identifiers[1:]:
                image_path = all_images[root][image_identifier]
                moving_image = sitk.ReadImage(image_path)
                moving_image = sitk.Resample(
                    moving_image,
                    target_image,
                    sitk.Transform(),
                    sitk.sitkLinear,
                    0,
                    moving_image.GetPixelID(),
                )
                sitk.WriteImage(moving_image, image_path)
