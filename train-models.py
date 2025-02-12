import re
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib
from scipy.stats import loguniform, uniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold, RFE, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from compare import get_gt

GT_PATH = "/data/barcode_1/Copy of BARCODE1 Top 10% Biopsy data download progeny 11.10.24 Final.xlsx"
AUX_DF = "/data/barcode_1/barcode1_metadata_cleaned.csv"
RADIOMIC_FEATURES = "data/radiomic_preds/preds.json"
CLASS_NAME = "Gleason score"
PI_RADS = "PIRADS score 1st read"
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
N_ITER_SEARCH = 50
N_JOBS = 16
SEED = 42
PATTERN = "DW.*_original_.*"

np.random.seed(SEED)

# Dictionary of models
models = {
    # "LightGBM": lgb.LGBMClassifier,
    "RandomForest": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    # "XGBoost": xgb.XGBClassifier
}

default_params = {
    "LightGBM": {"random_state": None, "verbosity": -1},
    "RandomForest": {"random_state": None, "class_weight": "balanced"},
    "LogisticRegression": {
        "penalty": 'elasticnet', 
        "solver": 'saga', 
        "max_iter": 10000, 
        "class_weight": "balanced",
        "random_state": None},
    "XGBoost": {"use_label_encoder": False, "eval_metric": "logloss", "random_state": None}
}

# Dictionary of hyperparameter search space (optimized for small dataset)
param_grids = {
    "LightGBM": {
        'num_leaves': randint(15, 31),
        'learning_rate': uniform(0.05, 0.1),
        'n_estimators': randint(10, 50)
    },
    "RandomForest": {
        'n_estimators': randint(50, 100),
        'max_depth': randint(5, 10),
        'min_samples_split': randint(2, 5)
    },
    "LogisticRegression": {
        'C': loguniform(0.01, 1),
        'l1_ratio': uniform(0.1, 0.5)
    },
    "XGBoost": {
        'n_estimators': randint(50, 100),
        'learning_rate': uniform(0.05, 0.1),
        'max_depth': randint(3, 6),
        'subsample': uniform(0.8, 1.0),
        'colsample_bytree': uniform(0.8, 1.0)
    }
}

def clean_features(data: list[dict]) -> list[dict]:
    return [d for d in data if d["features"] is not None]

def subset_features(data: list[dict], pattern: None) -> list[dict]:
    if pattern is None:
        return data
    for idx in range(len(data)):
        element = data[idx]
        element["features"] = {
            k: element["features"][k] for k in element["features"]
            if re.search(pattern, k) is not None
        }
        data[idx] = element
    return data

def feature_list_to_dataframe(data: list[dict]) -> pd.DataFrame:
    out = []
    for element in data:
        features = element["features"]
        features = {k: features[k]['0'] for k in features} # unpack features
        features["identifier"] = element["identifier"]
        out.append(features)
    return pd.DataFrame(out)

def hp_search(X: np.ndarray, y: np.ndarray) -> dict:
    outer_loop = StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=SEED)
    output = {}
    print(f"Total data size: {X.shape}")
    for fold_idx, (train_idxs, test_idxs) in enumerate(outer_loop.split(X,y)):
        train_X, train_y = X[train_idxs], y[train_idxs]
        test_X, test_y = X[test_idxs], y[test_idxs]
        output[fold_idx] = {}
        for model_key in models:
            def_params = default_params[model_key]
            if "random_state" in def_params:
                def_params["random_state"] = SEED
            params = {f"model__{k}": param_grids[model_key][k] for k in param_grids[model_key]}
            print(f"Testing model {model_key} in fold {fold_idx}")
            print(f"Training data size {X.shape}")
            model = models[model_key](**def_params)
            model = Pipeline(
                [
                    # ("nzv", VarianceThreshold()),
                    ("scaler", StandardScaler()),
                    # ("sfm", SelectFromModel(LinearSVC(C=0.1, penalty="l1", dual=False))),
                    # ("rfe", RFE(LinearSVC(max_iter=1000, penalty="l2", C=0.01), n_features_to_select=0.2, step=1)),
                    ("model", model)]
            )
            model = RandomizedSearchCV(
                model, 
                param_distributions=params, 
                n_iter=N_ITER_SEARCH, 
                n_jobs=N_JOBS,
                cv=StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=SEED),
                scoring="roc_auc",
                random_state=SEED,
                verbose=2,
            )
            model = model.fit(train_X, train_y)
            pred_y = model.predict(test_X)
            prob_y = model.predict_proba(test_X)
            output[fold_idx][model_key] = {
                "model": model,
                "train_X": train_X,
                "train_y": train_y,
                "test_X": test_X,
                "test_y": test_y,
                "pred_y": pred_y,
                "prob_y": prob_y,
            }
    return output

if __name__ == "__main__":
    gt = get_gt(GT_PATH)
    
    aux = pd.read_csv(AUX_DF)[["StudyInstanceUID", "PatientID"]].drop_duplicates()
    aux_gt = pd.merge(aux, gt, left_on="PatientID", right_on="Individual name")

    with open(RADIOMIC_FEATURES) as o:
        features = json.load(o)

    features = clean_features(features)
    features = subset_features(features, PATTERN)
    feature_df = feature_list_to_dataframe(features)
    feature_columns = [x for x in feature_df.columns if x != "identifier"]
    
    complete_df = pd.merge(
        feature_df, aux_gt, left_on="identifier", right_on="StudyInstanceUID")
    
    X = complete_df[feature_columns]
    y = complete_df["class"]

    model_hp_output = hp_search(X.to_numpy(), y.to_numpy())

    output = {
        "models": model_hp_output,
        "X": X,
        "y": y,
    }

    joblib.dump(output, "models.joblib")