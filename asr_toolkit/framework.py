import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Dict, Tuple
import jiwer

from .loss import CTCLoss, CrossEntropyLoss, RNNTLoss
from .text import TextProcess


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.cfg_model.optim.adamw)
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer, **self.cfg_model.lr_scheduler.one_cycle_lr
            ),
            "name": "lr_scheduler_logger",
        }
        return [optimizer], [lr_scheduler]

    def get_wer(
        self, targets: Tensor, inputs: Tensor, input_lengths: Tensor
    ) -> Tuple[List[str], List[str], float]:
        predict_sequences = self.recognize(inputs, input_lengths)
        label_sequences = list(map(self.text_process.int2text, targets))
        wer = torch.Tensor(
            [
                jiwer.wer(truth, hypot)
                for truth, hypot in zip(label_sequences, predict_sequences)
            ]
        )
        wer = torch.mean(wer).item()
        return label_sequences, predict_sequences, wer

    def log_output(self, predict, target, wer):
        print("=" * 50)
        print("Sample Predicts: ", predict)
        print("Sample Targets:", target)
        print("Mean WER:", wer)
        print("=" * 50)


class CTCModel(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        n_class: int,
        cfg_model: Dict,
        text_process: TextProcess,
        log_idx: int = 100,
    ):
        super().__init__()
        self.encoder = encoder
        self.out = nn.Linear(encoder.output_dim, n_class)
        self.criterion = CTCLoss(**cfg_model.loss.ctc)

        self.cfg_model = cfg_model
        self.text_process = text_process
        self.log_idx = log_idx
        self.save_hyperparameters()

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs, output_lengths = self.encoder(inputs, input_lengths)
        outputs = self.out(outputs)
        outputs = F.log_softmax(outputs, -1)
        return outputs, output_lengths

    @torch.no_grad()
    def decode(self, encoder_output: Tensor) -> str:
        encoder_output = encoder_output.unsqueeze(0)
        outputs = F.log_softmax(self.out(encoder_output), -1)
        argmax = outputs.squeeze(0).argmax(-1)
        return self.text_process.decode(argmax)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> List[str]:
        outputs = list()

        encoder_outputs, _ = self.encoder(inputs, input_lengths)

        for encoder_output in encoder_outputs:
            predict = self.decode(encoder_output)
            outputs.append(predict)

        return outputs

    def training_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        outputs, output_lengths = self(inputs, input_lengths)

        loss = self.criterion(
            outputs.permute(1, 0, 2), targets, output_lengths, target_lengths
        )

        self.log("train loss", loss)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        outputs, output_lengths = self(inputs, input_lengths)

        loss = self.criterion(
            outputs.permute(1, 0, 2), targets, output_lengths, target_lengths
        )

        self.log("test loss", loss)

        if batch_idx % self.log_idx == 0:
            label_sequences, predict_sequences, wer = self.get_wer(
                targets, inputs, input_lengths
            )
            self.log_output(predict_sequences[0], label_sequences[0], wer)
            self.log("test wer", wer)

        return loss

    def test_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        outputs, output_lengths = self(inputs, input_lengths)

        loss = self.criterion(
            outputs.permute(1, 0, 2), targets, output_lengths, target_lengths
        )

        self.log("test loss", loss)

        if batch_idx % self.log_idx == 0:
            label_sequences, predict_sequences, wer = self.get_wer(
                targets, inputs, input_lengths
            )
            self.log_output(predict_sequences[0], label_sequences[0], wer)
            self.log("test wer", wer)

        return loss


class AEDModel(BaseModel):
    """attention-based encoder-decoder"""

    """or use in transformer"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        n_class: int,
        cfg_model: Dict,
        text_process: TextProcess,
        log_idx: int = 100,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.out = nn.Linear(decoder.output_dim, n_class)
        self.criterion = CrossEntropyLoss(**cfg_model.loss.cross_entropy)

        self.cfg_model = cfg_model
        self.text_process = text_process
        self.log_idx = log_idx
        self.save_hyperparameters()

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ):
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs, hidden_state = self.decoder(targets, encoder_outputs)
        outputs = self.out(decoder_outputs)
        outputs = F.log_softmax(outputs, -1)
        return outputs

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_length: int) -> str:
        encoder_output = encoder_output.unsqueeze(0)
        targets = encoder_output.new_tensor(
            [self.text_process.sos_id], dtype=torch.int
        )

        last_token = -1
        hidden_state = None
        for i in range(max_length):
            
            targets = targets.unsqueeze(0)

            decoder_outputs, hidden_state = self.decoder(
                targets, encoder_output, hidden_state
            )
            outputs = F.log_softmax(self.out(decoder_outputs), -1)

            last_token = outputs.squeeze(0).argmax(-1)[-1]

            targets = torch.concat((targets.squeeze(0), last_token.unsqueeze(0)), -1)

            if last_token == self.text_process.eos_id:
                break

        return self.text_process.int2text(targets)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> List[str]:
        encoder_outputs, _ = self.encoder(inputs, input_lengths)

        outputs = list()
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            predict = self.decode(encoder_output, max_length)
            outputs.append(predict)

        return outputs

    def training_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        outputs = self(inputs, input_lengths, targets, target_lengths)

        bz, t, _ = outputs.size()
        outputs_edited = outputs.view(bz * t, -1)
        targets_edited = targets.to(dtype=torch.long).view(-1)
        loss = self.criterion(outputs_edited, targets_edited)

        self.log("train loss", loss)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        outputs = self(inputs, input_lengths, targets, target_lengths)

        bz, t, _ = outputs.size()
        outputs_edited = outputs.view(bz * t, -1)
        targets_edited = targets.to(dtype=torch.long).view(-1)
        loss = self.criterion(outputs_edited, targets_edited)

        self.log("test loss", loss)

        if batch_idx % self.log_idx == 0:
            label_sequences, predict_sequences, wer = self.get_wer(
                targets, inputs, input_lengths
            )
            self.log_output(predict_sequences[0], label_sequences[0], wer)
            self.log("test wer", wer)

        return loss

    def test_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        outputs = self(inputs, input_lengths, targets, target_lengths)

        bz, t, _ = outputs.size()
        outputs_edited = outputs.view(bz * t, -1)
        targets_edited = targets.to(dtype=torch.long).view(-1)
        loss = self.criterion(outputs_edited, targets_edited)

        self.log("test loss", loss)

        if batch_idx % self.log_idx == 0:
            label_sequences, predict_sequences, wer = self.get_wer(
                targets, inputs, input_lengths
            )
            self.log_output(predict_sequences[0], label_sequences[0], wer)
            self.log("test wer", wer)

        return loss


class RNNTModel(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        n_class: int,
        cfg_model: Dict,
        text_process: TextProcess,
        log_idx: int = 100,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.out = nn.Sequential(
            nn.Linear(encoder.output_dim + decoder.output_dim, encoder.output_dim),
            nn.Tanh(),
            nn.Linear(encoder.output_dim, n_class, bias=False),
        )
        self.criterion = RNNTLoss(**cfg_model.loss.rnnt)

        self.cfg_model = cfg_model
        self.text_process = text_process
        self.log_idx = log_idx
        self.save_hyperparameters()

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs, hidden_state = self.decoder(targets)

        outputs = self.joint(encoder_outputs, decoder_outputs)
        return outputs, encoder_output_lengths

    def joint(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        """
        Joint `encoder_outputs` and `decoder_outputs`.
        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            * outputs (torch.FloatTensor): outputs of joint `encoder_outputs` and `decoder_outputs`..
        """
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            input_length = encoder_outputs.size(1)
            target_length = decoder_outputs.size(1)

            encoder_outputs = encoder_outputs.unsqueeze(2)
            decoder_outputs = decoder_outputs.unsqueeze(1)

            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
            decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        outputs = self.out(outputs)
        outputs = F.log_softmax(outputs, -1)

        return outputs

    def get_batch(self, batch):
        inputs, input_lengths, targets, target_lengths = batch

        batch_size = inputs.size(0)

        zeros = torch.zeros((batch_size, 1)).to(device=self.device)
        compute_targets = torch.cat((zeros, targets), dim=1).to(
            device=self.device, dtype=torch.int
        )
        compute_target_lengths = (target_lengths + 1).to(device=self.device)

        return (
            inputs,
            input_lengths,
            targets,
            target_lengths,
            compute_targets,
            compute_target_lengths,
        )

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_length: int) -> str:
        """
        Decode `encoder_outputs`.
        Args:
            encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(seq_length, dimension)``
            max_length (int): max decoding time step
        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        pred_tokens, hidden_state = list(), None
        decoder_input = encoder_output.new_tensor(
            [[self.decoder.sos_id]], dtype=torch.int
        )

        for t in range(max_length):

            decoder_output, hidden_state = self.decoder(
                decoder_input, hidden_state=hidden_state
            )
            step_output = self.joint(
                encoder_output[t].view(-1), decoder_output.view(-1)
            )
            pred_token = step_output.argmax(dim=0)
            print(pred_token)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = step_output.new_tensor([[pred_token]], dtype=torch.long)

        pred_tokens = torch.LongTensor(pred_tokens)
        return self.text_process.int2text(pred_tokens)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> List[str]:
        """
        Recognize input speech. This method consists of the forward of the encoder and the decode() of the decoder.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        outputs = list()

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            decoded_seq = self.decode(encoder_output, max_length)
            outputs.append(decoded_seq)

        return outputs

    def training_step(self, batch: Tensor, batch_idx: int):
        (
            inputs,
            input_lengths,
            targets,
            target_lengths,
            compute_targets,
            compute_target_lengths,
        ) = self.get_batch(batch)

        outputs, output_lengths = self(
            inputs, input_lengths, compute_targets, compute_target_lengths
        )

        loss = self.criterion(outputs, targets, output_lengths, target_lengths)

        self.log("train loss", loss)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        (
            inputs,
            input_lengths,
            targets,
            target_lengths,
            compute_targets,
            compute_target_lengths,
        ) = self.get_batch(batch)

        outputs, output_lengths = self(
            inputs, input_lengths, compute_targets, compute_target_lengths
        )

        loss = self.criterion(outputs, targets, output_lengths, target_lengths)

        self.log("test loss", loss)

        if batch_idx % self.log_idx == 0:
            label_sequences, predict_sequences, wer = self.get_wer(
                targets, inputs, input_lengths
            )
            self.log_output(predict_sequences[0], label_sequences[0], wer)
            self.log("test wer", wer)

        return loss

    def test_step(self, batch: Tensor, batch_idx: int):
        (
            inputs,
            input_lengths,
            targets,
            target_lengths,
            compute_targets,
            compute_target_lengths,
        ) = self.get_batch(batch)

        outputs, output_lengths = self(
            inputs, input_lengths, compute_targets, compute_target_lengths
        )

        loss = self.criterion(outputs, targets, output_lengths, target_lengths)

        self.log("test loss", loss)

        if batch_idx % self.log_idx == 0:
            label_sequences, predict_sequences, wer = self.get_wer(
                targets, inputs, input_lengths
            )
            self.log_output(predict_sequences[0], label_sequences[0], wer)
            self.log("test wer", wer)

        return loss


class JointCTCAttentionModel(BaseModel):
    """attention-based encoder-decoder"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        n_class: int,
        ctc_weight: float,
        cfg_model: Dict,
        text_process: TextProcess,
        log_idx: int = 100,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_outputs = nn.Linear(encoder.output_dim, n_class)
        self.decoder_outputs = nn.Linear(decoder.output_dim, n_class)

        self.ctc_criterion = CTCLoss(**cfg_model.loss.ctc)
        self.ce_criterion = CrossEntropyLoss(**cfg_model.loss.cross_entropy)
        self.ctc_weight = ctc_weight
        self.attention_weight = 1 - self.ctc_weight

        self.cfg_model = cfg_model
        self.text_process = text_process
        self.log_idx = log_idx
        self.save_hyperparameters()

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
        hidden_state: Tensor = None,
    ) -> Tensor:
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs, hidden_state = self.decoder(
            targets, encoder_outputs, hidden_state
        )

        encoder_outputs = self.encoder_outputs(encoder_outputs)
        decoder_outputs = self.decoder_outputs(decoder_outputs)

        encoder_outputs = F.log_softmax(encoder_outputs, -1)
        decoder_outputs = F.log_softmax(decoder_outputs, -1)

        return encoder_outputs, encoder_output_lengths, decoder_outputs

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_length: int) -> str:
        encoder_output = encoder_output.unsqueeze(0)

        targets = encoder_output.new_tensor([self.text_process.sos_id], dtype=torch.int)

        last_token = -1
        hidden_state = None
        for i in range(max_length):
            if last_token == self.text_process.eos_id:
                break
            targets = targets.unsqueeze(0)

            decoder_outputs, hidden_state = self.decoder(
                targets, encoder_output, hidden_state
            )
            outputs = F.log_softmax(self.decoder_outputs(decoder_outputs), -1)

            last_token = outputs.squeeze(0).argmax(-1)[-1]

            targets = torch.concat((targets.squeeze(0), last_token.unsqueeze(0)), -1)

        return self.text_process.int2text(targets)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> List[str]:
        encoder_outputs, _ = self.encoder(inputs, input_lengths)

        outputs = list()
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            predict = self.decode(encoder_output, max_length)
            outputs.append(predict)

        return outputs

    def criterion(self, ctc_loss: Tensor, ce_loss: Tensor) -> Tensor:
        return self.ctc_weight * ctc_loss + self.attention_weight * ce_loss

    def training_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch

        encoder_outputs, encoder_output_lengths, decoder_outputs = self(
            inputs, input_lengths, targets, target_lengths
        )

        ctc_loss = self.ctc_criterion(
            encoder_outputs.permute(1, 0, 2),
            targets,
            encoder_output_lengths,
            target_lengths,
        )

        bz, t, _ = decoder_outputs.size()
        decoder_outputs_edited = decoder_outputs.view(bz * t, -1)
        targets_edited = targets.to(dtype=torch.long).view(-1)
        ce_loss = self.ce_criterion(decoder_outputs_edited, targets_edited)

        loss = self.criterion(ctc_loss, ce_loss)
        return self.text_process.int2text(targets)

    def validation_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch

        encoder_outputs, encoder_output_lengths, decoder_outputs = self(
            inputs, input_lengths, targets, target_lengths
        )

        ctc_loss = self.ctc_criterion(
            encoder_outputs.permute(1, 0, 2),
            targets,
            encoder_output_lengths,
            target_lengths,
        )

        bz, t, _ = decoder_outputs.size()
        decoder_outputs_edited = decoder_outputs.view(bz * t, -1)
        targets_edited = targets.to(dtype=torch.long).view(-1)
        ce_loss = self.ce_criterion(decoder_outputs_edited, targets_edited)

        loss = self.criterion(ctc_loss, ce_loss)

        self.log("test loss", loss)

        if batch_idx % self.log_idx == 0:
            label_sequences, predict_sequences, wer = self.get_wer(
                targets, inputs, input_lengths
            )
            self.log_output(predict_sequences[0], label_sequences[0], wer)
            self.log("test wer", wer)

        return loss

    def test_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch

        encoder_outputs, encoder_output_lengths, decoder_outputs = self(
            inputs, input_lengths, targets, target_lengths
        )

        ctc_loss = self.ctc_criterion(
            encoder_outputs.permute(1, 0, 2),
            targets,
            encoder_output_lengths,
            target_lengths,
        )

        bz, t, _ = decoder_outputs.size()
        decoder_outputs_edited = decoder_outputs.view(bz * t, -1)
        targets_edited = targets.to(dtype=torch.long).view(-1)
        ce_loss = self.ce_criterion(decoder_outputs_edited, targets_edited)

        loss = self.criterion(ctc_loss, ce_loss)

        self.log("test loss", loss)

        if batch_idx % self.log_idx == 0:
            label_sequences, predict_sequences, wer = self.get_wer(
                targets, inputs, input_lengths
            )
            self.log_output(predict_sequences[0], label_sequences[0], wer)
            self.log("test wer", wer)

        return loss
