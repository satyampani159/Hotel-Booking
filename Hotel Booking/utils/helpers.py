import os
import joblib
import pandas as pd
from typing import Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_model(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)


def load_model(path: str) -> Any:
    return joblib.load(path)
