import pandas as pd
from torch.utils.data import Dataset


class SentenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
    ):
        self.df = df

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return row['ru'], row['en']

    def __len__(self) -> int:
        return len(self.df)
