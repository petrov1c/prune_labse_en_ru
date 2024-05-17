from typing import List

from omegaconf import OmegaConf
from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str


class PruningConfig(BaseModel):
    pruning_ratio: float
    global_pruning: bool
    iterative_steps: int
    save_model: bool


class LossConfig(BaseModel):
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class DataConfig(BaseModel):
    debug: bool = False
    prepare_data: bool = False
    data_path: str
    batch_size: int
    n_workers: int
    train_size: float


class Config(BaseModel):
    project_name: str
    experiment_name: str
    tags: List[str]
    data_config: DataConfig
    n_epochs: int
    accelerator: str
    device: int
    monitor_metric: str
    monitor_mode: str
    model: ModelConfig
    pruning: PruningConfig
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    losses: List[LossConfig]

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
