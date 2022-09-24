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
        num_workers: int = 4,
        persistent_workers: bool = False,
    ):
        super().__init__()

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.predict_set = predict_set
        self.batch_size = batch_size

        self.text_process = text_process
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        self.save_hyperparameters(logger=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def tokenize(self, s):
        if isinstance(s, str):
            s = s.lower()
        s = self.text_process.tokenize(s)
        return s

    def _collate_fn(self, batch):
        """
        Take feature and input, transform and then padding it
        """

        specs = [i[0] for i in batch]
        input_lengths = torch.IntTensor([i.size(0) for i in specs])
        trans = [i[1] for i in batch]

        bs = len(specs)

        # batch, time, feature
        specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)

        trans = [self.text_process.text2int(self.tokenize(s)) for s in trans]
        target_lengths = torch.IntTensor([s.size(0) for s in trans])
        trans = torch.nn.utils.rnn.pad_sequence(trans, batch_first=True).to(
            dtype=torch.int
        )

        # concat sos and eos to transcript
        sos_id = torch.IntTensor([[self.text_process.sos_id]]).repeat(bs, 1)
        eos_id = torch.IntTensor([[self.text_process.eos_id]]).repeat(bs, 1)
        trans = torch.cat((sos_id, trans, eos_id), dim=1).to(dtype=torch.int)

        return specs, input_lengths, trans, target_lengths
