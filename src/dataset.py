from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.paths import FEATURE_COLS, FULL_CSV, TARGET_COL, TEST_CSV


def load_dataset(csv_path: Path | str | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path is not None else FULL_CSV
    df = pd.read_csv(path)
    missing = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLS].copy(), df[TARGET_COL].copy()


def encode_labels(
    y_train: pd.Series, y_test: pd.Series
) -> tuple[LabelEncoder, pd.Series, pd.Series]:
    encoder = LabelEncoder()
    y_train_enc = pd.Series(encoder.fit_transform(y_train), index=y_train.index, name=TARGET_COL)
    y_test_enc = pd.Series(encoder.transform(y_test), index=y_test.index, name=TARGET_COL)
    return encoder, y_train_enc, y_test_enc


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def export_test_csv(test_df: pd.DataFrame, path: Path | str | None = None) -> Path:
    out = Path(path) if path is not None else TEST_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    test_df[FEATURE_COLS + [TARGET_COL]].to_csv(out, index=False)
    return out
