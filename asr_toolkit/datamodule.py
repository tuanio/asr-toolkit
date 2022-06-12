import torch
from utils import TextProcess
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        trainset: Dataset,
        valset: Dataset,
        testset: Dataset,
        predict_set: Dataset,
        text_process: TextProcess,
        batch_size: int,
    ):
        super().__init__()

        self.trainset = trainset
        self.valset = testset
        self.testset = testset
        self.batch_size = batch_size

        self.text_process = text_process

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
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

    def _collate_fn(self, batch):
        """
        Take feature and input, transform and then padding it
        """
        specs = [i[0] for i in batch]
        input_lengths = torch.LongTensor([i.size(0) for i in specs])
        trans = [i[1] for i in batch]

        # batch, time, feature
        specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)
        specs = specs.unsqueeze(1)  # batch, channel, time, feature

        trans = [self.text_process.text2int(s) for s in trans]
        target_lengths = torch.LongTensor([s.size(0) for s in trans])
        trans = torch.nn.utils.rnn.pad_sequence(trans, batch_first=True).to(
            dtype=torch.long
        )

        return specs, input_lengths, trans, target_lengths
