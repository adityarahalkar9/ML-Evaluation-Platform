from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import torch

from app.core.utils import read_json, utc_now_iso, write_json_atomic
from app.ml.data import TabularPreprocessor
from app.ml.models import MLPClassifier, build_model


class ModelRegistry:
    """File-based model registry: one directory per model version.

    <model_id>/model.pt          weights + arch config (safe, tensors + primitives only)
    <model_id>/preprocessor.json fitted preprocessing state (JSON)
    <model_id>/metadata.json     provenance + metrics
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, model_id: str) -> Path:
        return self.root / model_id

    def exists(self, model_id: str) -> bool:
        return (self._dir(model_id) / "model.pt").is_file()

    def save(self, model: MLPClassifier, preprocessor: TabularPreprocessor, metadata: dict) -> str:
        model_id = f"mdl_{uuid.uuid4().hex[:10]}"
        model_dir = self._dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "format_version": 1,
            "input_dim": int(metadata["input_dim"]),
            "num_classes": int(metadata["num_classes"]),
            "model_config": metadata["model_config"],
            "classes": preprocessor.classes,
            "feature_names": preprocessor.feature_names,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }
        torch.save(checkpoint, model_dir / "model.pt")
        write_json_atomic(model_dir / "preprocessor.json", preprocessor.to_dict())
        write_json_atomic(model_dir / "metadata.json", {"model_id": model_id, "created_at": utc_now_iso(), **metadata})
        return model_id

    def load(self, model_id: str) -> tuple[MLPClassifier, TabularPreprocessor, dict]:
        if not self.exists(model_id):
            raise KeyError(model_id)
        ckpt = torch.load(self._dir(model_id) / "model.pt", map_location="cpu", weights_only=True)
        model = build_model(ckpt["model_config"], ckpt["input_dim"], ckpt["num_classes"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        preprocessor = TabularPreprocessor.from_dict(read_json(self._dir(model_id) / "preprocessor.json"))
        metadata = read_json(self._dir(model_id) / "metadata.json")
        return model, preprocessor, metadata

    def get_metadata(self, model_id: str) -> dict:
        path = self._dir(model_id) / "metadata.json"
        if not path.is_file():
            raise KeyError(model_id)
        return read_json(path)

    def list_models(self) -> list[dict]:
        out = []
        for meta_path in self.root.glob("*/metadata.json"):
            try:
                out.append(read_json(meta_path))
            except Exception:
                continue
        out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return out

    def delete(self, model_id: str) -> None:
        model_dir = self._dir(model_id)
        if not model_dir.exists():
            raise KeyError(model_id)
        shutil.rmtree(model_dir)