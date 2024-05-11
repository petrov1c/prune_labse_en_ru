import argparse
import logging
import os

import torch
import pytorch_lightning as pl
from clearml import OutputModel, Task
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import SentenceDM
from src.lightning_module import PruneModule
from src.pruning_module import prune_model


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    datamodule = SentenceDM(config.data_config)
    model = PruneModule(config)

    prune_model(model, datamodule)

    task = Task.init(
        project_name=config.project_name,
        task_name=f'{config.experiment_name}',
        auto_connect_frameworks=True,
    )
    task.connect(config.model_dump())

    experiment_save_path = os.path.join(EXPERIMENTS_PATH, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}',
    )
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        log_every_n_steps=20,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=config.monitor_metric, patience=4, mode=config.monitor_mode),
            LearningRateMonitor(logging_interval='epoch'),
        ],
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)

    output_model = OutputModel(task=task, name='latest')

    # Сохранение весов модели
    output_model.update_weights(weights_filename=checkpoint_callback.best_model_path, auto_delete_file=False)

    # ONNX версия
    # task.upload_artifact(name='ONNX model', artifact_object=onnx_checkpoint)


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)

    torch.set_float32_matmul_precision('high')
    pl.seed_everything(42, workers=True)
    config = Config.from_yaml(args.config_file)
    train(config)
