import torch
import torch.nn as nn
import math
from layers import PositionalEncoding, EncoderLayer, DecoderLayer


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

        self.emb_dim = emb_dim

        # 1. embedding layer
        self.encoder_embedding = nn.Embedding(src_vocab_size, emb_dim)
        self.decoder_embedding = nn.Embedding(target_vocab_size, emb_dim)

        # common parts for encoder and decoder
        self.positional_encoding = PositionalEncoding(emb_dim, max_length=max_length)
        self.dropout = nn.Dropout(p=dropout)

        # 2. encoder and decoder stacks
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(emb_dim, hidden_dim, n_heads, bias, dropout)
                for _ in range(n_layers)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(emb_dim, hidden_dim, n_heads, bias, dropout)
                for _ in range(n_layers)
            ]
        )

        # 3. classifier
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
        enc_emb = self.encoder_embedding(encoder_input) * math.sqrt(self.emb_dim)
        enc_emb = self.positional_encoding(enc_emb)
        enc_emb = self.dropout(enc_emb)

        encoder_output = enc_emb
        for layer in self.encoder_layers:
            encoder_output = layer(
                encoder_output, key_pad_mask=encoder_pad_mask, mode=mode
            )

        # --- Step 2. Decoder ---
        dec_emb = self.decoder_embedding(decoder_input) * math.sqrt(self.emb_dim)
        dec_emb = self.positional_encoding(dec_emb)
        dec_emb = self.dropout(dec_emb)

        decoder_output = dec_emb
        for layer in self.decoder_layers:
            decoder_output = layer(
                x=decoder_output,
                encoder_output=encoder_output,
                source_pad_mask=encoder_pad_mask,  # Cross-Attn용 (Encoder Padding)
                target_pad_mask=decoder_pad_mask,  # Self-Attn용 (Decoder Padding)
                mode=mode,
            )

        # --- Step 3. Output ---
        logits = self.classifier(decoder_output)
        return logits
