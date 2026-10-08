from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

MISSING_TOKEN = "__NA__"
NUMERIC_INFERENCE_THRESHOLD = 0.95
IDENTIFIER_MIN_ROWS = 100


def decode_csv_bytes(content: bytes) -> str:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode file as UTF-8 or Latin-1")


def load_csv_text(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        raise ValueError("CSV contains no rows")
    df.columns = [str(c).strip() for c in df.columns]
    dups = df.columns[df.columns.duplicated()].tolist()
    if dups:
        raise ValueError(f"Duplicate column names: {sorted(set(dups))}")
    return df


def infer_column_type(series: pd.Series) -> str:
    """'numeric' if >= 95% of non-null values parse as numbers, else 'categorical'."""
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    non_null = int(series.notna().sum())
    if non_null == 0:
        return "categorical"
    numeric = pd.to_numeric(series, errors="coerce")
    ratio = float(numeric.notna().sum()) / non_null
    return "numeric" if ratio >= NUMERIC_INFERENCE_THRESHOLD else "categorical"


def _opt_float(v) -> float | None:
    return None if pd.isna(v) else round(float(v), 4)


def column_profile(series: pd.Series, is_target: bool) -> dict:
    missing = int(series.isna().sum())
    unique = int(series.nunique(dropna=True))
    profile: dict[str, Any] = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "missing": missing,
        "missing_pct": round(100.0 * missing / max(len(series), 1), 2),
        "unique": unique,
    }
    if is_target:
        profile["role"] = "target"
        profile["class_distribution"] = {str(k): int(v) for k, v in series.value_counts().items()}
        return profile
    inferred = infer_column_type(series)
    profile["role"] = "feature"
    profile["inferred_type"] = inferred
    profile["likely_identifier"] = unique == len(series) and len(series) >= IDENTIFIER_MIN_ROWS
    if inferred == "numeric":
        s = pd.to_numeric(series, errors="coerce")
        profile["stats"] = {
            "min": _opt_float(s.min()), "max": _opt_float(s.max()),
            "mean": _opt_float(s.mean()), "median": _opt_float(s.median()),
            "std": _opt_float(s.std()),
        }
    else:
        profile["top_values"] = {str(k): int(v) for k, v in series.value_counts().head(10).items()}
    return profile


def profile_dataframe(df: pd.DataFrame, target: str) -> dict:
    return {
        "rows": int(len(df)),
        "columns": [column_profile(df[c], c == target) for c in df.columns],
    }


class TabularPreprocessor:
    """Column-level preprocessing fitted on the TRAINING SPLIT ONLY (no leakage).

    - numeric: median imputation + z-score standardization
    - categorical: "__NA__" fill + fixed-order one-hot; unseen values -> all-zero vector
    - target: string labels mapped to a fixed class index
    """

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str], classes: list[str] | None = None):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.classes = list(classes) if classes else []
        self.medians: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.categories: dict[str, list[str]] = {}
        self.feature_names: list[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame, target: str) -> "TabularPreprocessor":
        if not self.numeric_cols and not self.categorical_cols:
            raise ValueError("Preprocessor received no feature columns")
        y = df[target].astype(str).str.strip()
        self.classes = sorted(y.dropna().unique().tolist())

        for col in self.numeric_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            has_data = bool(s.notna().any())
            self.medians[col] = float(s.median()) if has_data else 0.0
            self.means[col] = float(s.mean()) if has_data else 0.0
            std = float(s.std()) if int(s.notna().sum()) > 1 else 0.0
            self.stds[col] = std if std > 0 else 1.0  # guard: constant columns

        for col in self.categorical_cols:
            s = df[col].fillna(MISSING_TOKEN).astype(str).str.strip()
            self.categories[col] = sorted(s.unique().tolist())

        self._build_feature_names()
        self._fitted = True
        return self

    def _build_feature_names(self) -> None:
        names = list(self.numeric_cols)
        for col in self.categorical_cols:
            names.extend(f"{col}={cat}" for cat in self.categories[col])
        self.feature_names = names

    @property
    def output_dim(self) -> int:
        return len(self.feature_names)

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        if not self._fitted:
            raise RuntimeError("Preprocessor is not fitted")
        required = self.numeric_cols + self.categorical_cols
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        warnings: list[str] = []

        blocks: list[np.ndarray] = []
        if self.numeric_cols:
            X_num = df[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
            X_num = X_num.fillna({c: self.medians[c] for c in self.numeric_cols})
            arr = X_num.to_numpy(dtype=np.float64)
            mu = np.array([self.means[c] for c in self.numeric_cols])
            sd = np.array([self.stds[c] for c in self.numeric_cols])
            blocks.append((arr - mu) / sd)

        for col in self.categorical_cols:
            cats = self.categories[col]
            index = {c: i for i, c in enumerate(cats)}
            s = df[col].fillna(MISSING_TOKEN).astype(str).str.strip()
            unseen = int((~s.isin(index)).sum())
            if unseen:
                warnings.append(f"column '{col}': {unseen} unseen value(s) encoded as all-zeros")
            codes = s.map(index)
            valid = codes.notna()
            onehot = np.zeros((len(df), len(cats)), dtype=np.float64)
            rows = np.flatnonzero(valid.to_numpy())
            vals = codes[valid].to_numpy(dtype=np.int64)
            onehot[rows, vals] = 1.0
            blocks.append(onehot)

        X = np.hstack(blocks).astype(np.float32) if blocks else np.zeros((len(df), 0), dtype=np.float32)
        return X, warnings

    def transform_target(self, values: pd.Series) -> np.ndarray:
        if values.isna().any():
            raise ValueError("Target column contains missing values")
        s = values.astype(str).str.strip()
        mapping = {c: i for i, c in enumerate(self.classes)}
        unknown = sorted(set(s) - set(mapping))
        if unknown:
            raise ValueError(f"Unknown target labels: {unknown}. Expected one of: {self.classes}")
        return s.map(mapping).to_numpy(dtype=np.int64)

    def to_dict(self) -> dict:
        return {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "classes": self.classes,
            "medians": self.medians,
            "means": self.means,
            "stds": self.stds,
            "categories": self.categories,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TabularPreprocessor":
        p = cls(d["numeric_cols"], d["categorical_cols"], d.get("classes"))
        p.medians = d.get("medians", {})
        p.means = d.get("means", {})
        p.stds = d.get("stds", {})
        p.categories = d.get("categories", {})
        p.feature_names = d.get("feature_names", [])
        p._fitted = True
        return p


def stratified_split_indices(
    y: np.ndarray, test_size: float, val_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class-stratified train/val/test split. Deterministic for a given (data, seed)."""
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        n_val = int(round(len(idx) * val_size))
        test_parts.append(idx[:n_test])
        val_parts.append(idx[n_test:n_test + n_val])
        train_parts.append(idx[n_test + n_val:])

    def _shuffle_concat(parts: list[np.ndarray]) -> np.ndarray:
        arr = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        rng.shuffle(arr)
        return arr

    return _shuffle_concat(train_parts), _shuffle_concat(val_parts), _shuffle_concat(test_parts)