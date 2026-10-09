from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    dataset_id: str
    batch_size: int = Field(512, ge=1, le=8192)
    warmup_batches: int = Field(2, ge=0, le=16)