import pandas as pd
import json
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, f1_score, recall_score

recall_score = precision_score

preds_path = "data/radiomic_preds/preds.json"
gt_path = "/data/barcode_1/Copy of BARCODE1 Top 10% Biopsy data download progeny 11.10.24 Final.xlsx"
aux_df = "/data/barcode_1/barcode1_metadata_cleaned.csv"

class_name = "Gleason score"
pi_rads = "PIRADS score 1st read"

def get_gt(gt_path: str) -> pd.DataFrame:
    gt = pd.read_excel(gt_path)
    gt["invisible"] = (gt[pi_rads] == 'PIRAD 1 (no lesion)') | (gt[pi_rads] == 'PIRAD 2 (no lesion)')
    gt = gt[["Individual name", class_name, "invisible", pi_rads]].drop_duplicates()
    gt["class"] = [eval(str(x)) if str(x) != "nan" else 0 for x in gt[class_name]]
    gt["class"] = gt["class"] > 6
    gt = (
        gt[["Individual name", "class", "invisible"]]
        .groupby("Individual name")
        .max()
        .reset_index())
    return gt

if __name__ == "__main__":
    gt = pd.read_excel(gt_path)
    gt["invisible"] = (gt[pi_rads] == 'PIRAD 1 (no lesion)') | (gt[pi_rads] == 'PIRAD 2 (no lesion)')
    gt = gt[["Individual name", class_name, "invisible", pi_rads]].drop_duplicates()
    gt["class"] = [eval(str(x)) if str(x) != "nan" else 0 for x in gt[class_name]]
    gt["class"] = gt["class"] > 6
    gt = (
        gt[["Individual name", "class", "invisible"]]
        .groupby("Individual name")
        .max()
        .reset_index())
    print(gt.shape)

    with open(preds_path) as o:
        preds = pd.DataFrame(json.load(o))
    
    aux = pd.read_csv(aux_df)[["StudyInstanceUID", "PatientID"]].drop_duplicates()
    aux_preds = pd.merge(preds, aux, left_on="identifier", right_on="StudyInstanceUID")

    aux_preds_gt = pd.merge(aux_preds, gt, left_on="PatientID", right_on="Individual name")

    print(aux_preds_gt.shape[0], aux_preds_gt["class"].sum())
    aux_preds_gt = aux_preds_gt.dropna(subset=["class", "y_prob"])
    print(aux_preds_gt.shape[0], aux_preds_gt["class"].sum())
    auc_score = roc_auc_score(aux_preds_gt["class"], aux_preds_gt["y_prob"])
    precision = precision_score(aux_preds_gt["class"], aux_preds_gt["y_pred"])
    BS = 1000
    random_distribution = np.sort([
        roc_auc_score(aux_preds_gt["class"].sample(aux_preds_gt["class"].shape[0]), aux_preds_gt["y_prob"])
        for i in range(BS)
    ])
    random_distribution_precision = np.sort([
        precision_score(aux_preds_gt["class"].sample(aux_preds_gt["class"].shape[0]), aux_preds_gt["y_pred"])
        for i in range(BS)
    ])
    cm = confusion_matrix(aux_preds_gt["class"], aux_preds_gt["y_pred"])
    print("AUC:", auc_score, 1 - sum(auc_score > random_distribution) / BS)
    print("Precision:", precision, 1 - sum(precision > random_distribution_precision) / BS)
    print("CM:", cm)

    for tf in [False, True]:
        print("Invisible:", tf)
        idxs = aux_preds_gt["invisible"] == tf
        sub_class = aux_preds_gt["class"][idxs]
        sub_prob = aux_preds_gt["y_prob"][idxs]
        sub_pred = aux_preds_gt["y_pred"][idxs]
        auc_score = roc_auc_score(sub_class, sub_prob)
        cm = confusion_matrix(sub_class, sub_pred)
        precision = precision_score(sub_class, sub_pred)
        random_distribution = np.sort([
            roc_auc_score(sub_class.sample(sub_pred.shape[0]), sub_prob)
            for i in range(BS)
        ])
        random_distribution_precision = np.sort([
            precision_score(sub_class.sample(sub_pred.shape[0]), sub_pred)
            for i in range(BS)
        ])
        print("AUC:", auc_score, 1 - sum(auc_score > random_distribution) / BS)
        print("CM:", cm)
        print("Precision:", precision, 1 - sum(precision > random_distribution_precision) / BS)

