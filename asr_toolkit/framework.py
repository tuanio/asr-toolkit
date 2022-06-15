import torch
from torch import nn, Tensor, optim
import pytorch_lightning as pl
from typing import List, Dict
from loss import CTCLoss, CrossEntropyLoss, RNNTLoss

class CTC(pl.LightningModule):
    def __init__(self, encoder: nn.Module, lr: float, cfg: Dict):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        self.criterion = CTCLoss(**cfg.loss)
        self.save_hyperparameters()
    
    def forward(self, inputs, input_lengths):
        # encoder recognize (logits tensor or inputs and input length)
        return self.encoder.recognize(inputs, input_lengths)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, **self.cfg.optim)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch
        outputs = self.encoder(inputs, input_lengths)
        loss = self.criterion(
            outputs.permute(1, 0, 2), targets, input_lengths, target_lengths
        )

        self.log("train_loss", loss)
        self.log("lr", self.lr)

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch
        outputs = self.encoder(inputs, input_lengths)
        loss = self.criterion(
            outputs.permute(1, 0, 2), targets, input_lengths, target_lengths
        )

    def test_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch
        outputs = self.encoder(inputs, input_lengths)
        loss = self.criterion(
            outputs.permute(1, 0, 2), targets, input_lengths, target_lengths
        )
