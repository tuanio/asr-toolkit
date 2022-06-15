import torch
from torch import nn, Tensor, optim
import pytorch_lightning as pl
from typing import List, Dict
from loss import CTCLoss, CrossEntropyLoss, RNNTLoss
from text import TextProcess
import jiwer
from typing import Tuple, List


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, **self.cfg.optim)
        return optimizer

    def get_wer(
        self, targets: Tensor, outputs: Tensor
    ) -> Tuple[List[str], List[str], float]:
        argmax = outputs.argmax(-1)
        label_sequences = [self.text_process.int2text(sent) for sent in targets]
        predict_sequences = [self.text_process.decode(sent) for sent in argmax]
        wer = [
            jiwer.wer(truth, hypot)
            for truth, hypot in zip(label_sequences, predict_sequences)
        ]
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
        self, encoder: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        self.criterion = CTCLoss(**cfg.loss.ctc)
        self.text_process = text_process
        self.log_idx = log_idx
        self.save_hyperparameters()

    def forward(self, inputs, input_lengths):
        # encoder recognize (logits tensor or inputs and input length)
        output = self.encoder(inputs, input_lengths)
        predict = self.text_process.decode(output.argmax(-1))
        return predict

    def training_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        outputs, output_lengths = self.encoder(inputs, input_lengths)
        loss = self.criterion(
            outputs.permute(1, 0, 2), targets, output_lengths, target_lengths
        )

        self.log("train loss", loss)
        self.log("lr", self.lr)

    def validation_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        outputs, output_lengths = self.encoder(inputs, input_lengths)
        loss = self.criterion(
            outputs.permute(1, 0, 2), targets, output_lengths, target_lengths
        )

        label_sequences, predict_sequences, wer = self.get_wer(targets, outputs)

        self.log("validation loss", loss)
        self.log("validation wer", wer)

        if batch_idx % self.log_idx == 0:
            self.log_output(predict_sequences[0], label_sequences[0], wer)

    def test_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        outputs, output_lengths = self.encoder(inputs, input_lengths)
        loss = self.criterion(
            outputs.permute(1, 0, 2), targets, output_lengths, target_lengths
        )

        label_sequences, predict_sequences, wer = self.get_wer(targets, outputs)

        self.log("test loss", loss)
        self.log("test wer", wer)

        if batch_idx % self.log_idx == 0:
            self.log_output(predict_sequences[0], label_sequences[0], wer)


class AEDModel(BaseModel):
    """attention-based encoder-decoder"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float,
        cfg: Dict,
        text_process: TextProcess,
        log_idx: int = 100,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cfg = cfg
        self.criterion = CrossEntropyLoss(**cfg.loss.cross_entropy)
        self.lr = lr
        self.text_process = text_process
        self.log_idx = log_idx
        self.save_hyperparameters()

    def training_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs = self.decoder(targets, target_lengths, encoder_outputs)

        loss = self.criterion(decoder_outputs, targets)

        self.log("train loss", loss)
        self.log("lr", self.lr)

    def validation_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs = self.decoder(targets, target_lengths, encoder_outputs)

        loss = self.criterion(decoder_outputs, targets)

        label_sequences, predict_sequences, wer = self.get_wer(targets, decoder_outputs)

        self.log("validation loss", loss)
        self.log("validation wer", wer)

        if batch_idx % self.log_idx == 0:
            self.log_output(predict_sequences[0], label_sequences[0], wer)

    def test_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs = self.decoder(targets, target_lengths, encoder_outputs)

        loss = self.criterion(decoder_outputs, targets)

        label_sequences, predict_sequences, wer = self.get_wer(targets, decoder_outputs)

        self.log("test loss", loss)
        self.log("test wer", wer)

        if batch_idx % self.log_idx == 0:
            self.log_output(predict_sequences[0], label_sequences[0], wer)


class RNNTModel(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float,
        cfg: Dict,
        text_process: TextProcess,
        log_idx: int = 100,
    ):
        self.encoder = encoder
        self.decoder = decoder

        self.cfg = cfg
        self.criterion = RNNTLoss(**cfg.loss.rnnt)
        self.lr = lr
        self.text_process = text_process
        self.log_idx = log_idx
        self.save_hyperparameters()

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
        outputs = self.fc(outputs)

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
    def decode(self, encoder_output: Tensor, max_length: int) -> Tensor:
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
            [[self.decoder.sos_id]], dtype=torch.long
        )

        for t in range(max_length):

            decoder_output, hidden_state = self.decoder(
                decoder_input, hidden_states=hidden_state
            )
            step_output = self.joint(
                encoder_output[t].view(-1), decoder_output.view(-1)
            )
            step_output = step_output.softmax(dim=0)
            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = step_output.new_tensor([[pred_token]], dtype=torch.long)

        return torch.LongTensor(pred_tokens)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor):
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

        outputs = torch.stack(outputs, dim=1).transpose(0, 1)

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

        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs = self.decoder(targets, target_lengths)

        outputs = self.joint(encoder_outputs, decoder_outputs)
        output_lengths = encoder_output_lengths

        loss = self.criterion(outputs, targets, output_lengths, target_lengths)

        self.log("train loss", loss)

    def validation_step(self, batch: Tensor, batch_idx: int):
        (
            inputs,
            input_lengths,
            targets,
            target_lengths,
            compute_targets,
            compute_target_lengths,
        ) = self.get_batch(batch)

        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs = self.decoder(targets, target_lengths)

        outputs = self.joint(encoder_outputs, decoder_outputs)
        output_lengths = encoder_output_lengths

        loss = self.criterion(outputs, targets, output_lengths, target_lengths)

        label_sequences, predict_sequences, wer = self.get_wer(targets, decoder_outputs)

        self.log("validation loss", loss)
        self.log("validation wer", wer)

        if batch_idx % self.log_idx == 0:
            self.log_output(predict_sequences[0], label_sequences[0], wer)

    def test_step(self, batch: Tensor, batch_idx: int):
        (
            inputs,
            input_lengths,
            targets,
            target_lengths,
            compute_targets,
            compute_target_lengths,
        ) = self.get_batch(batch)

        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs = self.decoder(targets, target_lengths)

        outputs = self.joint(encoder_outputs, decoder_outputs)
        output_lengths = encoder_output_lengths

        loss = self.criterion(outputs, targets, output_lengths, target_lengths)

        label_sequences, predict_sequences, wer = self.get_wer(targets, decoder_outputs)

        self.log("test loss", loss)
        self.log("test wer", wer)

        if batch_idx % self.log_idx == 0:
            self.log_output(predict_sequences[0], label_sequences[0], wer)

    def get_wer(
        self, targets: Tensor, inputs: Tensor, input_lengths: Tensor
    ) -> Tuple[List[str], List[str], float]:
        predict_sequences = self.recognize(inputs, input_lengths)
        predict_sequences = [
            self.text_process.int2text(sent) for sent in predict_sequences
        ]
        label_sequences = [self.text_process.int2text(sent) for sent in targets]
        wer = [
            jiwer.wer(truth, hypot)
            for truth, hypot in zip(label_sequences, predict_sequences)
        ]
        wer = torch.mean(wer).item()
        return label_sequences, predict_sequences, wer


class JointCTCAttentionModel(BaseModel):
    """attention-based encoder-decoder"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        ctc_lambda: float,
        lr: float,
        cfg: Dict,
        text_process: TextProcess,
        log_idx: int = 100,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cfg = cfg
        self.ctc_criterion = CTCLoss(**cfg.loss.ctc)
        self.ce_criterion = CrossEntropyLoss(**cfg.loss.cross_entropy)
        self.ctc_lambda = ctc_lambda
        self.lr = lr
        self.text_process = text_process
        self.log_idx = log_idx
        self.save_hyperparameters()

    def criterion(self, ctc_loss: Tensor, ce_loss: Tensor) -> Tensor:
        return self.ctc_lambda * ctc_loss + (1 - self.ctc_lambda) * ce_loss

    def training_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs = self.decoder(targets, target_lengths, encoder_outputs)

        ctc_loss = self.ctc_criterion(
            encoder_outputs.permute(1, 0, 2),
            targets,
            encoder_output_lengths,
            target_lengths,
        )
        ce_loss = self.ce_criterion(decoder_outputs, targets)
        loss = self.criterion(ctc_loss, ce_loss)

        self.log("train loss", loss)
        self.log("lr", self.lr)

    def validation_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs = self.decoder(targets, target_lengths, encoder_outputs)

        ctc_loss = self.ctc_criterion(
            encoder_outputs.permute(1, 0, 2),
            targets,
            encoder_output_lengths,
            target_lengths,
        )
        ce_loss = self.ce_criterion(decoder_outputs, targets)
        loss = self.criterion(ctc_loss, ce_loss)

        label_sequences, predict_sequences, wer = self.get_wer(targets, decoder_outputs)

        self.log("validation loss", loss)
        self.log("validation wer", wer)

        if batch_idx % self.log_idx == 0:
            self.log_output(predict_sequences[0], label_sequences[0], wer)

    def test_step(self, batch: Tensor, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs = self.decoder(targets, target_lengths, encoder_outputs)

        ctc_loss = self.ctc_criterion(
            encoder_outputs.permute(1, 0, 2),
            targets,
            encoder_output_lengths,
            target_lengths,
        )
        ce_loss = self.ce_criterion(decoder_outputs, targets)
        loss = self.criterion(ctc_loss, ce_loss)

        label_sequences, predict_sequences, wer = self.get_wer(targets, decoder_outputs)

        self.log("test loss", loss)
        self.log("test wer", wer)

        if batch_idx % self.log_idx == 0:
            self.log_output(predict_sequences[0], label_sequences[0], wer)
