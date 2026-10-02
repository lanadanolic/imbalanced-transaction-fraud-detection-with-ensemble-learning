from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(data_path: Path) -> pd.DataFrame:

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}"
        )

    return pd.read_csv(data_path)


def split_features_target(
    df: pd.DataFrame,
    target_column: str = "Class"
):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def split_dataset(
    X,
    y,
    test_size: float = 0.20,
    random_state: int = 42
):

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def scale_features(
    X_train,
    X_test
):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def apply_smote(
    X_train,
    y_train,
    random_state: int = 42
):

    smote = SMOTE(
        random_state=random_state
    )

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    return X_train_resampled, y_train_resampled