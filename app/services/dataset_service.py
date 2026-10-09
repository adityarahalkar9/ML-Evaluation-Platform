from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd

from app.core.exceptions import AppError, NotFoundError
from app.core.utils import read_json, sha256_bytes, utc_now_iso, write_json_atomic
from app.ml import data as ml_data


class DatasetService:
    """Persists uploaded CSVs + profiles under storage/datasets/<dataset_id>/."""

    def __init__(self, root: Path, experiments_root: Path, max_rows: int, max_classes: int, max_mb: float):
        self.root = root
        self.experiments_root = experiments_root
        self.max_rows = max_rows
        self.max_classes = max_classes
        self.max_mb = max_mb
        self.root.mkdir(parents=True, exist_ok=True)

    def ingest(self, filename: str, content: bytes, target_column: str) -> dict:
        if not content:
            raise AppError("Uploaded file is empty", 422, "empty_file")
        if len(content) > self.max_mb * 1024 * 1024:
            raise AppError(f"File exceeds {self.max_mb} MB limit", 413, "file_too_large")

        text = ml_data.decode_csv_bytes(content)
        df = ml_data.load_csv_text(text)
        target_column = target_column.strip()
        if target_column not in df.columns:
            raise AppError(
                f"Target column '{target_column}' not found. Columns: {list(df.columns)}",
                422, "invalid_target",
            )

        warnings: list[str] = []
        dropped = int(df[target_column].isna().sum())
        if dropped:
            df = df[df[target_column].notna()].reset_index(drop=True)
            warnings.append(f"dropped {dropped} row(s) with missing target")
        if df.empty:
            raise AppError("No usable rows after removing rows with missing target", 422, "empty_file")
        if len(df) > self.max_rows:
            raise AppError(f"Dataset has {len(df)} rows; max is {self.max_rows}", 413, "file_too_large")

        target_str = df[target_column].astype(str).str.strip()
        classes = sorted(target_str.unique().tolist())
        if len(classes) < 2:
            raise AppError("Target must contain at least 2 distinct classes", 422, "invalid_target")
        if len(classes) > self.max_classes:
            raise AppError(f"Target has {len(classes)} classes; max is {self.max_classes}", 422, "invalid_target")

        profile = ml_data.profile_dataframe(df, target_column)
        for col in profile["columns"]:
            if col.get("likely_identifier"):
                warnings.append(
                    f"column '{col['name']}' looks like an identifier (all unique); "
                    "consider adding it to exclude_cols when training"
                )

        dataset_id = f"dset_{uuid.uuid4().hex[:10]}"
        ds_dir = self.root / dataset_id
        ds_dir.mkdir(parents=True)
        (ds_dir / "source.csv").write_bytes(content)      # provenance (original upload)
        df.to_csv(ds_dir / "data.csv", index=False)       # cleaned copy used for training

        record = {
            "dataset_id": dataset_id,
            "filename": filename,
            "target_column": target_column,
            "created_at": utc_now_iso(),
            "checksum_sha256": sha256_bytes(content),
            "rows": int(len(df)),
            "num_columns": int(df.shape[1]),
            "classes": classes,
            "class_distribution": {str(k): int(v) for k, v in target_str.value_counts().items()},
            "warnings": warnings,
            "profile": profile,
        }
        write_json_atomic(ds_dir / "dataset.json", record)
        return record

    def get(self, dataset_id: str) -> dict:
        path = self.root / dataset_id / "dataset.json"
        if not path.is_file():
            raise NotFoundError("Dataset", dataset_id)
        return read_json(path)

    def list(self) -> list[dict]:
        out = []
        for p in self.root.glob("*/dataset.json"):
            try:
                rec = read_json(p)
                out.append({
                    "dataset_id": rec["dataset_id"],
                    "filename": rec.get("filename"),
                    "rows": rec.get("rows"),
                    "target_column": rec.get("target_column"),
                    "classes": rec.get("classes"),
                    "created_at": rec.get("created_at"),
                })
            except Exception:
                continue
        out.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return out

    def load_dataframe(self, dataset_id: str) -> pd.DataFrame:
        self.get(dataset_id)  # raises 404 if unknown
        df = pd.read_csv(self.root / dataset_id / "data.csv")
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def delete(self, dataset_id: str, force: bool = False) -> dict:
        self.get(dataset_id)
        if not force:
            refs = self._referencing_experiments(dataset_id)
            if refs:
                raise AppError(
                    f"Dataset is referenced by {len(refs)} experiment(s): {refs[:5]}. "
                    "Pass force=true to delete anyway.",
                    409, "dataset_in_use",
                )
        shutil.rmtree(self.root / dataset_id)
        return {"deleted": dataset_id}

    def _referencing_experiments(self, dataset_id: str) -> list[str]:
        refs: list[str] = []
        if not self.experiments_root.exists():
            return refs
        for p in self.experiments_root.glob("*/experiment.json"):
            try:
                rec = read_json(p)
                if rec.get("dataset", {}).get("dataset_id") == dataset_id:
                    refs.append(rec.get("experiment_id", p.parent.name))
            except Exception:
                continue
        return refs