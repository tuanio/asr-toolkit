import torch
from torch import nn, Tensor
from torchaudio import transforms as T


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.loss(input, target)


class CTCLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.CTCLoss(**kwargs)

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        return self.loss(log_probs, targets, input_lengths, target_lengths)


class RNNTLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = T.RNNTLoss(**kwargs)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        logit_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        return self.loss(logits, targets, logit_lengths, target_lengths)
