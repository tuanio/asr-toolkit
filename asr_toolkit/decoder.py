import torch
from torch import nn, Tensor
from typing import Tuple


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        n_class: int,
        encoder_output_dim: int,
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
        lstm_output_dim = hidden_size + hidden_size * bidirectional
        self.output_proj = nn.Linear(lstm_output_dim, encoder_output_dim)
        self.output_dim = encoder_output_dim

        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                self.output_dim, num_heads=num_attention_heads, batch_first=True
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
        outputs = self.output_proj(outputs)

        # output: (batch size, seq len, self.output_dim)
        if self.use_attention:
            attn_output, attn_output_weights = self.attention(
                outputs, encoder_outputs, encoder_outputs
            )
            outputs = attn_output
        return outputs


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_class: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-05,
        batch_first: bool = True,
        norm_first: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_class, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
        )
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)
        self.output_dim = d_model

    def forward(
        self, targets: Tensor, target_lengths: Tensor, encoder_outputs: Tensor
    ) -> Tensor:
        embedded = self.embedding(targets)
        outputs = self.decoder(tgt=embedded, memory=encoder_outputs)
        return outputs
