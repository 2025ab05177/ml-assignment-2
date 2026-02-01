import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    y_pred: np.ndarray
    y_proba: np.ndarray | None


def compute_metrics_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    average: str = "macro",
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    out["Accuracy"] = float(accuracy_score(y_true, y_pred))
    out["Precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    out["Recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    out["F1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    out["MCC"] = float(matthews_corrcoef(y_true, y_pred))

    # AUC requires probabilities; for multi-class, use OvR
    if y_proba is not None:
        try:
            out["AUC"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average=average))
        except Exception:
            out["AUC"] = float("nan")
    else:
        out["AUC"] = float("nan")

    return out


def ensure_artifacts_dir(path: str = "artifacts") -> None:
    import os
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_predict_proba(model, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


def make_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def make_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] | None = None,
) -> str:
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )