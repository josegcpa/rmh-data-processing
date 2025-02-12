import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, f1_score, recall_score

models = joblib.load("models.joblib")

results = {}

for fold in models["models"]:
    for model in models["models"][fold]:
        prob_y = models["models"][fold][model]['prob_y']
        pred_y = models["models"][fold][model]['pred_y']
        test_y = models["models"][fold][model]['test_y']
        metrics = {
            "auc": roc_auc_score(test_y, prob_y[:,1]),
            "recall": recall_score(test_y, pred_y),
            "precision": precision_score(test_y, pred_y),
            "confusion_matrix": confusion_matrix(test_y, pred_y),
            "n_positive": test_y.sum(),
            "n_total": test_y.shape[0],
        }
        if model not in results:
            results[model] = {k: [] for k in metrics}
        for k in metrics:
            results[model][k].append(metrics[k])

for model in results:
    for k in results[model]:
        print(model, k, sum(results[model][k]) / len(results[model][k]))