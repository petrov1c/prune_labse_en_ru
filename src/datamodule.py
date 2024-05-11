import logging
import os
from typing import Optional

import csv
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.config import DataConfig
from src.dataset import SentenceDataset
from sklearn.model_selection import train_test_split


class SentenceDM(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.cfg = config
        self.debug = self.cfg.debug

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self):
        if self.train_dataset is None:
            split_and_save_datasets(self.cfg.data_path, self.cfg.train_size)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            df_train = read_df(self.cfg.data_path, 'train')
            df_valid = read_df(self.cfg.data_path, 'valid')
            self.train_dataset = SentenceDataset(df_train, self.debug)
            self.valid_dataset = SentenceDataset(df_valid, self.debug)

        elif stage == 'test':
            df_test = read_df(self.cfg.data_path, 'test')
            self.test_dataset = SentenceDataset(df_test, self.debug)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.n_workers,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.n_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.n_workers,
            shuffle=False,
            drop_last=False,
        )


def split_and_save_datasets(data_path: str, train_fraction: float = 0.8):
    df_en = pd.read_csv(
        os.path.join(data_path, '1mcorpus/corpus.en_ru.1m.en'),
        sep='\t',
        header=None,
        quoting=csv.QUOTE_NONE,
    )
    df_ru = pd.read_csv(
        os.path.join(data_path, '1mcorpus/corpus.en_ru.1m.ru'),
        sep='\t',
        header=None,
        quoting=csv.QUOTE_NONE,
    )
    df = pd.DataFrame({'ru': df_ru[0], 'en': df_en[0]})
    logging.info('Original dataset: {0}'.format(len(df)))

    train_df, else_df = train_test_split(df, train_size=train_fraction, shuffle=True)
    test_df, valid_df = train_test_split(else_df, train_size=0.5, shuffle=True)

    logging.info('Train dataset: {0}'.format(len(train_df)))
    logging.info('Valid dataset: {0}'.format(len(valid_df)))
    logging.info('Test dataset: {0}'.format(len(test_df)))

    train_df.to_csv(os.path.join(data_path, 'df_train.csv'), index=False)
    valid_df.to_csv(os.path.join(data_path, 'df_valid.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'df_test.csv'), index=False)
    logging.info('Datasets successfully saved!')


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_path, f'df_{mode}.csv'))
