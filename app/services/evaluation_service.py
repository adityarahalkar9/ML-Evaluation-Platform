from __future__ import annotations

import uuid
from pathlib import Path

from app.core.exceptions import AppError
from app.core.utils import utc_now_iso, write_json_atomic
from app.ml import data as ml_data
from app.ml.metrics import classification_report
from app.services.inference_service import InferenceService


class EvaluationService:
    def __init__(self, inference: InferenceService, datasets, evaluations_root: Path):
        self.inference = inference
        self.datasets = datasets
        self.evaluations_root = evaluations_root
        self.evaluations_root.mkdir(parents=True, exist_ok=True)

    def evaluate_dataset(self, model_id: str, dataset_id: str, batch_size: int, warmup_batches: int) -> dict:
        dataset = self.datasets.get(dataset_id)
        df = self.datasets.load_dataframe(dataset_id)
        return self._evaluate(model_id, df, dataset["target_column"], batch_size, warmup_batches, dataset_id)

    def evaluate_file(self, model_id: str, filename: str, content: bytes, target_column: str,
                      batch_size: int, warmup_batches: int) -> dict:
        if not target_column:
            raise AppError("target_column is required when evaluating an uploaded file", 422, "invalid_target")
        try:
            text = ml_data.decode_csv_bytes(content)
            df = ml_data.load_csv_text(text)
        except ValueError as exc:
            raise AppError(str(exc), 422, "invalid_file")
        record = self._evaluate(model_id, df, target_column, batch_size, warmup_batches, None)
        record["source_file"] = filename
        return record

    def _evaluate(self, model_id: str, df, target: str, batch_size: int, warmup_batches: int,
                  dataset_id: str | None) -> dict:
        predictor = self.inference.get_predictor(model_id)
        if target not in df.columns:
            raise AppError(f"Target column '{target}' not present in the data", 422, "invalid_target")
        df = df[df[target].notna()].reset_index(drop=True)
        try:
            y_true = predictor.pre.transform_target(df[target])
        except ValueError as exc:
            raise AppError(str(exc), 422, "invalid_target")

        out = predictor.predict(df, batch_size=batch_size, warmup_batches=warmup_batches)
        report = classification_report(y_true, out["predictions"], predictor.classes, probs=out["probabilities"])
        report["latency"] = out["latency"]

        record = {
            "evaluation_id": f"eval_{uuid.uuid4().hex[:10]}",
            "created_at": utc_now_iso(),
            "model_id": model_id,
            "dataset_id": dataset_id,
            "num_rows": out["num_rows"],
            "classes": predictor.classes,
            "device": predictor.device.type,
            "batch_size": int(batch_size),
            "warmup_batches": int(warmup_batches),
            "metrics": report,
        }
        write_json_atomic(self.evaluations_root / record["evaluation_id"] / "evaluation.json", record)
        return record