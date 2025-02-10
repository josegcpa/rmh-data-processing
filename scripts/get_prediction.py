import os
import argparse
import re
import json
import joblib
import mlflow
from tqdm import tqdm

from pathlib import Path
from radiomic_utils import (
    load_image,
    load_mask,
    itk_to_sitk,
    coregister,
    radiomics_extraction,
)


def main(config, loaded_model, preprocessing_pipeline, pdt):
    t2w_path = config["t2_series"]
    dwi_path = config["dwi_series"]
    adc_path = config["adc_series"]
    t2w, moving_image = load_image(t2w_path)
    dwi, fixed_image = load_image(dwi_path)
    adc, _ = load_image(adc_path)

    t2w_mask, moving_mask = load_mask(config["t2w_mask_path"])

    transform_mask = coregister(fixed_image, moving_image, moving_mask)
    dwi_mask = itk_to_sitk(transform_mask)
    adc_mask = itk_to_sitk(transform_mask)

    final = radiomics_extraction(t2w, t2w_mask, dwi, dwi_mask, adc, adc_mask)
    volume = final.loc[0, "T2W_original_shape_VoxelVolume"]

    final = preprocessing_pipeline.transform(final.loc[0, :].to_frame().T)
    y_prob = loaded_model.predict_proba(final)[:, 1][0]
    y_pred = int((y_prob > pdt) * 1)

    if y_pred == 1:
        message = f"The model predicted aggressive cancer, with a probability of aggressiveness of {y_prob}, given a probability decision threshold of {pdt}."
    else:
        message = f"The model predicted non-aggressive cancer, with a probability of aggressiveness of {y_prob}, given a probability decision threshold of {pdt}."

    return {
        "model": "1vs2345_112023",
        "y_prob": y_prob,
        "y_pred": y_pred,
        "message": message,
        "volume": volume,
        "error": None,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="gland_radiomics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--mask_folder", type=str, required=True)
    parser.add_argument("--t2_pattern", type=str, required=True)
    parser.add_argument("--dwi_pattern", type=str, required=True)
    parser.add_argument("--adc_pattern", type=str, required=True)
    parser.add_argument("--mask_pattern", type=str, required=True)
    parser.add_argument("--id_pattern", type=str, required=True)
    parser.add_argument("--model_folder", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    args = parser.parse_args()
    config = vars(args)

    data_dict = {}
    for image in Path(args.image_folder).glob("*"):
        image = str(image)
        if re.search(args.id_pattern, image):
            identifier = re.search(args.id_pattern, image).group()
            if identifier not in data_dict:
                data_dict[identifier] = {}
        else:
            continue
        if re.search(args.t2_pattern, image):
            data_dict[identifier]["t2_series"] = str(image)
        elif re.search(args.dwi_pattern, image):
            data_dict[identifier]["dwi_series"] = str(image)
        elif re.search(args.adc_pattern, image):
            data_dict[identifier]["adc_series"] = str(image)

    for mask in Path(args.mask_folder).glob("*"):
        mask = str(mask)
        if re.search(args.id_pattern, mask):
            identifier = re.search(args.id_pattern, mask).group()
            if identifier not in data_dict:
                data_dict[identifier] = {}
        else:
            continue
        if re.search(args.mask_pattern, mask):
            data_dict[identifier]["t2w_mask_path"] = str(mask)

    data_dict = {k: data_dict[k] for k in data_dict if len(data_dict[k]) == 4}

    model_folder = args.model_folder
    loaded_model = mlflow.sklearn.load_model(model_folder)
    preprocessing_pipeline = joblib.load(
        os.path.join(model_folder, "fitted_preprocessing_pipeline.pkl")
    )
    f = open(os.path.join(model_folder, "pdt.txt"), "r")
    pdt = float(f.readline())

    all_out = []
    for k in tqdm(data_dict):
        # potentially: change how this is done to store files individually
        # and check if they exist before running
        try:
            out = main(data_dict[k], loaded_model, preprocessing_pipeline, pdt)
        except Exception as e:
            out = {
                "model": "1vs2345_112023",
                "y_prob": None,
                "y_pred": None,
                "message": "",
                "volume": None,
                "error": str(e),
            }
        out["identifier"] = k
        all_out.append(out)

    with open(args.output_path, "w") as f:
        json.dump(all_out, f)
