import torch
import torch.nn as nn
import math
from layers import Encoder, Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        target_vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        bias: bool = True,
        dropout: float = 0.1,
        max_length: int = 5000,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            bias=bias,
            dropout=dropout,
            max_length=max_length,
        )
        self.decoder = Decoder(
            vocab_size=target_vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            bias=bias,
            dropout=dropout,
            max_length=max_length,
        )

        self.classifier = nn.Linear(in_features=emb_dim, out_features=target_vocab_size)

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_pad_mask: torch.Tensor | None = None,
        decoder_pad_mask: torch.Tensor | None = None,
        mode: str = "post-norm",
    ):
        """
        Args:
            encoder_input: source input ids (batch_size, seq_len)
            decoder_input: target input ids (batch_size, seq_len)
            encoder_pad_mask: padding mask for source input (batch_size, seq_len)
            decoder_pad_mask: padding mask for target input (batch_size, seq_len)
            mode: "pre-norm" or "post-norm" -> normalization mode for the encoder and decoder layers
        Return:
            logits: logits of the classifier (batch_size, seq_len, vocab_size)
        """
        # --- Step 1. Encoder ---
        encoder_output = self.encoder(
            x=encoder_input, key_pad_mask=encoder_pad_mask, mode=mode
        )

        # --- Step 2. Decoder ---
        decoder_output = self.decoder(
            x=decoder_input,
            encoder_output=encoder_output,
            source_pad_mask=encoder_pad_mask,
            target_pad_mask=decoder_pad_mask,
            mode=mode,
        )

        # --- Step 3. Output ---
        logits = self.classifier(decoder_output)
        return logits
