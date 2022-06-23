import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from asr_toolkit.text import TextProcess


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_set: Dataset,
        val_set: Dataset,
        test_set: Dataset,
        predict_set: Dataset = None,
        text_process: TextProcess = None,
        batch_size: int = 4,
    ):
        super().__init__()

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.predict_set = predict_set
        self.batch_size = batch_size

        self.text_process = text_process
        self.batch_size = batch_size

        self.save_hyperparameters(logger=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def tokenize(self, s):
        s = s.lower()
        s = ["<s>"] + self.text_process.tokenize(s) + ["<e>"]
        return s

    def _collate_fn(self, batch):
        """
        Take feature and input, transform and then padding it
        """
        specs = [i[0] for i in batch]
        input_lengths = torch.IntTensor([i.size(0) for i in specs])
        trans = [i[1] for i in batch]

        # batch, time, feature
        specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)

        trans = [self.text_process.text2int(self.tokenize(s)) for s in trans]
        target_lengths = torch.IntTensor([s.size(0) for s in trans])
        trans = torch.nn.utils.rnn.pad_sequence(trans, batch_first=True).to(
            dtype=torch.int
        )

        return specs, input_lengths, trans, target_lengths
