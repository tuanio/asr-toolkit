import torch
from torch import nn, Tensor
from typing import Tuple


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        n_class: int,
        hidden_size: int,
        num_layers: int,
        use_attention: bool = False,
        num_attention_heads: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_class, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        )
        self.output_dim = hidden_size + hidden_size * bidirectional
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                self.output_dim, num_heads=num_attention_heads
            )

    def forward(
        self, targets: Tensor, target_lengths: Tensor, encoder_outputs: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
            input
                targets: batch of sequence label integer
                target_lengths: batch of length of each sequence
                encoder_outputs (optional): output of encoder
                    -> (batch size, seq len, output_dim)
        """
        embedded = self.embedding(targets)
        outputs, (h, c) = self.lstm(embedded)
        # output: (batch size, seq len, self.output_dim)
        if self.use_attention:
            attn_output, attn_output_weights = self.attention(
                outputs, encoder_outputs, encoder_outputs
            )
            outputs = attn_output
        return outputs, target_lengths
