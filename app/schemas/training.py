from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelParams(BaseModel):
    """MLP architecture."""
    model_config = ConfigDict(protected_namespaces=())  # allow field named "model" upstream

    hidden_layers: list[int] = [64, 32]
    dropout: float = Field(0.1, ge=0.0, le=0.9)
    use_batchnorm: bool = True


class TrainingParams(BaseModel):
    epochs: int = Field(40, ge=1, le=500)
    batch_size: int = Field(64, ge=1, le=4096)
    learning_rate: float = Field(1e-3, gt=0.0, le=1.0)
    weight_decay: float = Field(0.0, ge=0.0, le=1.0)
    early_stopping_patience: int = Field(10, ge=1, le=100)
    early_stopping_min_delta: float = Field(1e-4, ge=0.0)
    use_class_weights: bool = False
    val_size: float = Field(0.15, gt=0.0, lt=0.5)
    test_size: float = Field(0.2, gt=0.0, lt=0.5)
    seed: int = Field(42, ge=0)

    @model_validator(mode="after")
    def check_split(self):
        if self.val_size + self.test_size >= 0.8:
            raise ValueError("val_size + test_size must be < 0.8")
        return self


class TrainingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    dataset_id: str
    model: ModelParams = ModelParams()
    training: TrainingParams = TrainingParams()
    exclude_cols: list[str] = []