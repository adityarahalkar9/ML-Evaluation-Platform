from typing import Any

from pydantic import BaseModel, Field


class PredictRowsRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1)
    batch_size: int = Field(512, ge=1, le=8192)


class PredictionRowOut(BaseModel):
    row_index: int
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    model_id: str
    classes: list[str]
    num_rows: int
    source_file: str | None = None
    predictions: list[PredictionRowOut]
    warnings: list[str]
    latency: dict[str, Any]