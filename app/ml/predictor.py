from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from app.ml.data import TabularPreprocessor
from app.ml.metrics import latency_stats_ms


class Predictor:
    """Serves a registered model. Raw DataFrame in -> predictions + latency stats out.

    Batched forward passes; warm-up batches are run untimed first (lazy init,
    cudnn autotune) so reported percentiles reflect steady-state serving.
    """

    def __init__(self, model_id: str, model, preprocessor: TabularPreprocessor, metadata: dict, device: torch.device):
        self.model_id = model_id
        self.model = model
        self.pre = preprocessor
        self.metadata = metadata
        self.device = device
        self.model.to(device)
        self.model.eval()

    @property
    def classes(self) -> list[str]:
        return self.pre.classes

    def predict(self, df: pd.DataFrame, batch_size: int = 512, warmup_batches: int = 0) -> dict:
        t0 = time.perf_counter()
        X, warnings = self.pre.transform(df)
        preprocess_ms = (time.perf_counter() - t0) * 1000.0
        n = len(X)

        if n == 0:
            return {
                "predictions": np.array([], dtype=np.int64),
                "probabilities": np.zeros((0, len(self.classes)), dtype=np.float32),
                "latency": {"count": 0, "batch_size": batch_size, "num_batches": 0,
                            "rows_per_second": None, "device": self.device.type,
                            "total_preprocess_ms": round(preprocess_ms, 3), "total_inference_ms": 0.0},
                "warnings": warnings, "num_rows": 0,
            }

        batch_times: list[float] = []
        pred_chunks: list[np.ndarray] = []
        prob_chunks: list[np.ndarray] = []

        with torch.no_grad():
            for bi, start in enumerate(range(0, n, batch_size)):
                xb = torch.from_numpy(X[start:start + batch_size]).to(self.device)
                if bi < warmup_batches:
                    self.model(xb)  # untimed warm-up pass
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                t_start = time.perf_counter()
                logits = self.model(xb)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()  # GPU work is async; sync for honest timing
                batch_times.append((time.perf_counter() - t_start) * 1000.0)
                prob_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
                pred_chunks.append(logits.argmax(dim=1).cpu().numpy())

        y_pred = np.concatenate(pred_chunks)
        probs = np.vstack(prob_chunks)
        infer_total = sum(batch_times)

        latency = latency_stats_ms(batch_times)
        latency.update({
            "batch_size": int(batch_size),
            "num_batches": len(batch_times),
            "total_preprocess_ms": round(preprocess_ms, 3),
            "total_inference_ms": round(infer_total, 3),
            "rows_per_second": round(n / (infer_total / 1000.0), 1) if infer_total > 0 else None,
            "device": self.device.type,
        })
        return {"predictions": y_pred, "probabilities": probs, "latency": latency,
                "warnings": warnings, "num_rows": n}

    def predict_payload(self, df: pd.DataFrame, batch_size: int = 512) -> dict:
        """API-friendly output: per-row label, confidence, full probability dict."""
        out = self.predict(df, batch_size=batch_size)
        idx_to_class = {i: c for i, c in enumerate(self.classes)}
        rows = []
        for i in range(out["num_rows"]):
            prob_row = out["probabilities"][i]
            pred_i = int(out["predictions"][i])
            rows.append({
                "row_index": i,
                "predicted_class": idx_to_class[pred_i],
                "confidence": round(float(prob_row[pred_i]), 6),
                "probabilities": {idx_to_class[j]: round(float(p), 6) for j, p in enumerate(prob_row)},
            })
        return {
            "model_id": self.model_id,
            "classes": self.classes,
            "num_rows": out["num_rows"],
            "predictions": rows,
            "warnings": out["warnings"],
            "latency": out["latency"],
        }