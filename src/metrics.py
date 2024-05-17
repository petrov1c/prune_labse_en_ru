import torch
from torchmetrics import MeanSquaredError, MetricCollection
from torchmetrics import Metric


class BenchmarkAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('mean_s', default=torch.tensor(0))

    def update(self, *args, **kwargs) -> None:
        return None

    def compute(self) -> torch.Tensor:
        return self.mean_s


def get_metrics(**kwargs) -> MetricCollection:
    return MetricCollection(
        {
            'mse': MeanSquaredError(**kwargs),
            'mean_s': BenchmarkAccuracy(**kwargs),
        },
    )
