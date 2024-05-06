from torchmetrics import MeanSquaredError, MetricCollection


def get_metrics(**kwargs) -> MetricCollection:
    return MetricCollection(
        {
            'mse': MeanSquaredError(**kwargs),
        }
    )
