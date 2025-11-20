import torch
import torch.nn as nn
import math
from layers import PositionalEncoding, Encoder


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        bias: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.emb_dim = emb_dim

        # 1. embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, emb_dim)
        self.positional_encoding = PositionalEncoding(emb_dim, max_length=5000)
        self.dropout = nn.Dropout(p=dropout)

        # encoder layers stack
        self.encoder_layers = nn.ModuleList(
            [
                Encoder(emb_dim, hidden_dim, n_heads, bias, dropout)
                for _ in range(n_layers)
            ]
        )

        # classifier
        self.classifier = nn.Linear(in_features=emb_dim, out_features=vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: None | torch.Tensor = None,
        mode: str = "post-norm",
        **kwargs: dict,
    ):
        """
        Args:
            x: input ids (batch_size, seq_len)
            pad_mask: padding mask (batch_size, seq_len)
            mode: "pre-norm" or "post-norm" -> normalization mode for the encoder layers
        Return:
            logits: logits of the classifier (batch_size, seq_len, vocab_size)
        """
        # 1. embedding  & positional encoding: (batch_size, seq_len, emb_dim)
        embedded = self.embedding_layer(x)
        # scale the embedded tensor by the square root of the embedding dimension
        # -> to stabilize the training process
        embedded = embedded * math.sqrt(self.emb_dim)
        embedded = self.positional_encoding(embedded)
        embedded = self.dropout(embedded)

        # 2. pass through encoder layers stack: (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, emb_dim)
        for layer in self.encoder_layers:
            embedded = layer(embedded, pad_mask=pad_mask, mode=mode)

        # 3. classifier: (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, vocab_size)
        logits = self.classifier(embedded)
        return logits
