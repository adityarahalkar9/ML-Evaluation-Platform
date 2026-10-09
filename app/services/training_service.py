from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from app.core.exceptions import AppError
from app.core.jobs import Job, JobManager
from app.core.seed import enable_determinism, set_global_seed
from app.core.utils import read_json, utc_now_iso, write_json_atomic
from app.ml import data as ml_data
from app.ml.metrics import classification_report
from app.ml.models import build_model
from app.ml.predictor import Predictor
from app.ml.registry import ModelRegistry
from app.ml.trainer import train_model


class TrainingService:
    def __init__(self, datasets, registry: ModelRegistry, jobs: JobManager,
                 experiments_root: Path, device_resolver: Callable, deterministic: bool):
        self.datasets = datasets
        self.registry = registry
        self.jobs = jobs
        self.experiments_root = experiments_root
        self.device_resolver = device_resolver
        self.deterministic = deterministic
        self.experiments_root.mkdir(parents=True, exist_ok=True)

    def submit_job(self, request) -> dict:
        dataset_record = self.datasets.get(request.dataset_id)  # fail fast (404) if unknown
        job_id = self.jobs.submit(
            "training",
            f"train model on dataset {request.dataset_id}",
            lambda job: self._run(job, request, dataset_record),
        )
        return {"job_id": job_id, "status": "queued", "dataset_id": request.dataset_id}

    def _run(self, job: Job, request, dataset_record: dict) -> dict:
        job.set_progress(0.05, "loading dataset")
        df = self.datasets.load_dataframe(request.dataset_id)
        target = dataset_record["target_column"]
        df = df[df[target].notna()].reset_index(drop=True)

        # 1. feature columns
        unknown = [c for c in (request.exclude_cols or []) if c not in df.columns]
        if unknown:
            raise AppError(f"exclude_cols not present in dataset: {unknown}", 422, "invalid_columns")
        exclude = set(request.exclude_cols or []) | {target}
        feature_cols = [c for c in df.columns if c not in exclude]
        if not feature_cols:
            raise AppError("No feature columns left after exclusions", 422, "invalid_columns")
        numeric_cols = [c for c in feature_cols if ml_data.infer_column_type(df[c]) == "numeric"]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]
        job.log(f"features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

        # 2. labels + stratified split BEFORE fitting the preprocessor (no leakage)
        set_global_seed(request.training.seed)
        if self.deterministic:
            enable_determinism()
        device = self.device_resolver()
        job.log(f"device: {device}")

        y_full = df[target].astype(str).str.strip()
        classes = sorted(y_full.unique().tolist())
        y = y_full.map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=np.int64)

        tr, va, te = ml_data.stratified_split_indices(
            y, request.training.test_size, request.training.val_size, request.training.seed
        )
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            raise AppError("Dataset too small for the requested val/test split sizes", 422, "split_error")
        job.log(f"split: train={len(tr)} val={len(va)} test={len(te)} (stratified, seed={request.training.seed})")

        job.set_progress(0.10, "preprocessing")
        preprocessor = ml_data.TabularPreprocessor(numeric_cols, categorical_cols)
        preprocessor.fit(df.iloc[tr], target)
        preprocessor.classes = classes  # full class list (rare labels may be absent from a tiny train split)
        X_train, _ = preprocessor.transform(df.iloc[tr])
        X_val, _ = preprocessor.transform(df.iloc[va])
        X_test, _ = preprocessor.transform(df.iloc[te])
        y_train, y_val, y_test = y[tr], y[va], y[te]

        # 3. model + training loop
        job.set_progress(0.15, "building model")
        model_config = request.model.model_dump()
        training_config = request.training.model_dump()
        model = build_model(model_config, preprocessor.output_dim, len(classes))
        job.log(f"model: MLP input_dim={preprocessor.output_dim} "
                f"hidden={model_config['hidden_layers']} classes={len(classes)}")

        train_out = train_model(
            model, X_train, y_train, X_val, y_val,
            num_classes=len(classes),
            epochs=training_config["epochs"],
            batch_size=training_config["batch_size"],
            learning_rate=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            early_stopping_patience=training_config["early_stopping_patience"],
            early_stopping_min_delta=training_config["early_stopping_min_delta"],
            use_class_weights=training_config["use_class_weights"],
            device=device,
            seed=training_config["seed"],
            progress_cb=lambda frac, step: job.set_progress(0.15 + 0.65 * frac, step),
            log_cb=job.log,
        )
        job.log(f"training finished in {train_out['duration_sec']}s "
                f"(best epoch {train_out['best_epoch']}/{train_out['epochs_run']})")

        # 4. test evaluation through the SAME serving path used at inference time
        job.set_progress(0.85, "evaluating on held-out test split")
        predictor = Predictor("(unsaved)", model, preprocessor, {}, device)
        test_out = predictor.predict(df.iloc[te], batch_size=512, warmup_batches=2)
        report = classification_report(y_test, test_out["predictions"], classes, probs=test_out["probabilities"])
        report["latency"] = test_out["latency"]

        # 5. persist model + reproducible experiment record
        job.set_progress(0.92, "saving model and experiment record")
        experiment_id = f"exp_{uuid.uuid4().hex[:10]}"
        metadata = {
            "input_dim": preprocessor.output_dim,
            "num_classes": len(classes),
            "model_config": model_config,
            "experiment_id": experiment_id,
            "dataset_id": request.dataset_id,
            "dataset_name": dataset_record["filename"],
            "training_config": training_config,
            "feature_cols": {"numeric": numeric_cols, "categorical": categorical_cols},
            "seed": training_config["seed"],
            "device_trained": str(device),
            "torch_version": torch.__version__,
            "val_metrics": train_out["best_val_metrics"],
            "test_metrics": _slim_metrics(report),
            "training_duration_sec": train_out["duration_sec"],
        }
        model_id = self.registry.save(model, preprocessor, metadata)

        experiment = {
            "experiment_id": experiment_id,
            "created_at": utc_now_iso(),
            "status": "completed",
            "dataset": {
                "dataset_id": request.dataset_id,
                "name": dataset_record["filename"],
                "checksum_sha256": dataset_record.get("checksum_sha256"),
                "rows": dataset_record.get("rows"),
                "classes": classes,
            },
            "config": {
                "model": model_config,
                "training": training_config,
                "exclude_cols": request.exclude_cols,
                "feature_cols": {"numeric": numeric_cols, "categorical": categorical_cols},
            },
            "config_hash": _config_hash(request),
            "environment": _environment_info(device),
            "split": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
            "training": train_out,
            "test_metrics": report,
            "model_id": model_id,
        }
        write_json_atomic(self.experiments_root / experiment_id / "experiment.json", experiment)

        job.set_progress(1.0, "done")
        job.log(f"model saved: {model_id} | experiment: {experiment_id}")
        return {
            "job_id": job.id,
            "model_id": model_id,
            "experiment_id": experiment_id,
            "best_epoch": train_out["best_epoch"],
            "epochs_run": train_out["epochs_run"],
            "duration_sec": train_out["duration_sec"],
            "val_metrics": train_out["best_val_metrics"],
            "test_metrics": _slim_metrics(report),
        }

    # -- experiments ------------------------------------------------------
    def list_experiments(self) -> list[dict]:
        summaries = []
        for p in self.experiments_root.glob("*/experiment.json"):
            try:
                rec = read_json(p)
                summaries.append({
                    "experiment_id": rec["experiment_id"],
                    "created_at": rec.get("created_at"),
                    "dataset_id": rec.get("dataset", {}).get("dataset_id"),
                    "model_id": rec.get("model_id"),
                    "config_hash": rec.get("config_hash"),
                    "test_metrics": _slim_metrics(rec.get("test_metrics", {})),
                })
            except Exception:
                continue
        summaries.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return summaries

    def get_experiment(self, experiment_id: str) -> dict:
        path = self.experiments_root / experiment_id / "experiment.json"
        if not path.is_file():
            from app.core.exceptions import NotFoundError
            raise NotFoundError("Experiment", experiment_id)
        return read_json(path)


def _slim_metrics(report: dict) -> dict:
    return {
        "accuracy": report.get("accuracy"),
        "f1_macro": report.get("macro_avg", {}).get("f1"),
        "f1_weighted": report.get("weighted_avg", {}).get("f1"),
        "roc_auc_macro": report.get("roc_auc_macro"),
    }


def _config_hash(request) -> str:
    payload = {
        "model": request.model.model_dump(),
        "training": request.training.model_dump(),
        "exclude_cols": sorted(request.exclude_cols or []),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _environment_info(device) -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": __import__("pandas").__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }