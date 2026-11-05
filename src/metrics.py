import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ArrayLike = pd.DataFrame | pd.Series | NDArray[np.float64] | NDArray[np.int64]
LabelLike = pd.Series | NDArray[np.int64]


def evaluate_classifier(
    model: ClassifierMixin,
    X: ArrayLike,
    y_true: LabelLike,
    *,
    average: str = "weighted",
) -> dict[str, float]:
    y_pred = model.predict(X)

    metrics: dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
        metrics["AUC"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average=average))
    else:
        metrics["AUC"] = float("nan")

    return metrics


def metrics_table(all_metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    order = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    frame = pd.DataFrame(all_metrics).T
    return frame[order].round(4)


def confusion_and_report(
    model: ClassifierMixin,
    X: ArrayLike,
    y_true: LabelLike,
    class_names: list[str] | NDArray[np.str_],
) -> tuple[NDArray[np.int64], str]:
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(class_names),
        zero_division=0,
    )
    return cm, report
