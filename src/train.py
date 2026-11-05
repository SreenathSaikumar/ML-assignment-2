import json
from pathlib import Path
from typing import TypedDict

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.dataset import (
    encode_labels,
    export_test_csv,
    features_and_target,
    load_dataset,
    stratified_split,
)
from src.metrics import evaluate_classifier, metrics_table
from src.paths import MODEL_DIR, MODEL_FILES


class TrainResult(TypedDict):
    metrics: dict[str, dict[str, float]]
    table: pd.DataFrame
    class_names: list[str]
    train_size: int
    test_size: int
    model_dir: Path


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=12,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Naive Bayes": Pipeline([("clf", GaussianNB())]),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        SVC(
                            kernel="rbf",
                            C=1.0,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                        ensemble=False,
                    ),
                ),
            ]
        ),
    }


def train_and_save(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    model_dir: Path | None = None,
) -> TrainResult:
    model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    train_df, test_df = stratified_split(df, test_size=test_size, random_state=random_state)
    export_test_csv(test_df)

    X_train, y_train = features_and_target(train_df)
    X_test, y_test = features_and_target(test_df)
    encoder, y_train_enc, y_test_enc = encode_labels(y_train, y_test)

    joblib.dump(encoder, model_dir / "label_encoder.joblib")

    all_metrics: dict[str, dict[str, float]] = {}
    models = build_models(random_state=random_state)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train_enc)
        filename = MODEL_FILES[name]
        joblib.dump(pipe, model_dir / filename)
        all_metrics[name] = evaluate_classifier(pipe, X_test, y_test_enc)

    table = metrics_table(all_metrics)
    metrics_path = model_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    table.to_csv(model_dir / "metrics_table.csv")

    return {
        "metrics": all_metrics,
        "table": table,
        "class_names": list(encoder.classes_),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "model_dir": model_dir,
    }


if __name__ == "__main__":
    result = train_and_save()
    print(result["table"].to_string())
    print(f"\nSaved models to {result['model_dir']}")
    print(f"Train={result['train_size']} Test={result['test_size']}")
