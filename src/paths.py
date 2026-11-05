from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
MODEL_DIR: Path = ROOT / "model"
FULL_CSV: Path = DATA_DIR / "sensor_readings_24.csv"
TEST_CSV: Path = DATA_DIR / "test_data.csv"

FEATURE_COLS: list[str] = [f"US{i}" for i in range(1, 25)]
TARGET_COL: str = "class"

MODEL_FILES: dict[str, str] = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
