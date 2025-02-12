import os
import argparse
import re
import json
import joblib
import numpy as np
import pandas as pd
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

def to_json_serializable(x):
    if isinstance(x, dict):
        for k in x:
            x[k] = to_json_serializable(x[k])
    if isinstance(x, list):
        x = [to_json_serializable(y) for y in x]
    if isinstance(x, np.ndarray):
        if x.shape == []:
            x = float(x)
        else:
            x = x.tolist()
    return x

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

    rad_features = radiomics_extraction(t2w, t2w_mask, dwi, dwi_mask, adc, adc_mask)
    volume = rad_features.loc[0, "T2W_original_shape_VoxelVolume"]

    return {
        "model": "1vs2345_112023",
        "y_prob": None,
        "y_pred": None,
        "pdt": pdt,
        "volume": float(volume),
        "features": to_json_serializable(rad_features.to_dict()),
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
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    for k in tqdm(data_dict):
        # potentially: change how this is done to store files individually
        # and check if they exist before running
        out_path = os.path.join(args.output_path, f"{k}.json")
        if os.path.exists(out_path) is False:
            try:
                out = main(data_dict[k], loaded_model, preprocessing_pipeline, pdt)
            except Exception as e:
                out = {
                    "model": "1vs2345_112023",
                    "y_prob": None,
                    "y_pred": None,
                    "pdt": pdt,
                    "volume": None,
                    "features": None,
                    "error": str(e),
                }
                print(f"{k} failed with error {str(e)}")
            out["identifier"] = k
            with open(out_path, "w") as o:
                json.dump(out, o)
        else:
            with open(out_path, "r") as o:
                out = json.load(o)
        all_out.append(out)

    for idx in tqdm(range(len(all_out))):
        out = all_out[idx]
        rad_features = pd.DataFrame(out["features"])
        if rad_features is None:
            continue
        elif rad_features.shape[0] == 0:
            continue
        final = preprocessing_pipeline.transform(rad_features)
        print(rad_features.shape, final.shape)
        y_prob = loaded_model.predict_proba(final)[:, 1][0]
        y_pred = int((y_prob > pdt) * 1)
        out["y_prob"] = y_prob
        out["y_pred"] = y_pred
        all_out[idx] = out

    with open(os.path.join(args.output_path, "preds.json"), "w") as f:
        json.dump(all_out, f)
