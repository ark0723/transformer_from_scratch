import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        n_heads: int,
        n_layers: int,
        bias: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads."

        self.head_dim = emb_dim // n_heads
        # set tokenizer
        self.n_heads = n_heads
        self.n_layers = n_layers
        # layer initialization
        self.embedding_layer = nn.Embedding(vocab_size, emb_dim)

        # w_q, w_k, w_v initialization
        self.w_q = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=bias)
        self.w_k = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=bias)
        self.w_v = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=bias)

        # final context vector
        self.w_o = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # classifier: linear layer to map context vector to logits
        self.classifier = nn.Linear(in_features=emb_dim, out_features=vocab_size)

    def self_attention(self, x: torch.Tensor):
        """
        Args:
            x: embedded tensor (batch_size, seq_len, emb_dim)
        """
        # Q, K, V computation
        q = self.w_q(x)  # (Batch, Seq, Dim)
        k = self.w_k(x)  # (Batch, Seq, Dim)
        v = self.w_v(x)  # (Batch, Seq, Dim)

        # 2. Attention Score computation
        # k.transpose(-2, -1): (Batch, Dim, Seq)
        # torch.bmm : only applicable for 3-dimensional tensors
        # -> for this model, we will use torch.matmul for multi-head attention(batch, head, seq, dim) : 4-dimensional tensors required
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.emb_dim)

        # 3. Softmax
        attn_weights = F.softmax(scores, dim=-1)
        # apply dropout
        attn_weights = self.dropout(attn_weights)

        # 4. Context Vector
        context_vector = torch.matmul(attn_weights, v)

        return context_vector

    def multi_head_attention(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: embedded tensor (batch_size, seq_len, emb_dim)
        """

        batch_size, seq_len, emb_dim = x.size()
        # 1. Q, K, V computation : shape (batch_size, seq_len, emb_dim)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2. Q, K, V split by heads: shape (batch_size, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # attention masking
        if mask is not None:
            # input mask shape : (batch_size, seq_len)
            # mask shape change required for broadcasting : (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            # masked_fill : fill the values in attn_scores where 0 (padding position) with -inf
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        # apply softmax and dropout
        # shape (batch_size, n_heads, seq_len, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 3. context vector
        # shape (batch_size, n_heads, seq_len, head_dim)
        context_vector_by_heads = torch.matmul(attn_weights, v)

        # 4. concatenate context vectors by heads
        context_vector = (
            context_vector_by_heads.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.n_heads * self.head_dim)
        )
        context_vector = self.w_o(
            context_vector
        )  # shape (batch_size, seq_len, emb_dim)

        return context_vector

    def forward(self, x, mask=None):
        """
        Args:
            x: input ids (batch_size, seq_len)
        Return:
            attn_output: context vector (batch_size, seq_len, emb_dim)
        """
        # 1. embedding
        embedded = self.embedding_layer(x)  # (batch_size, seq_len, emb_Dim)
        # 2. multi-head attention -> (batch_size, seq_len, emb_Dim)
        attn_output = self.multi_head_attention(embedded, mask=mask)

        # 3. logits calculation : (batch_size, seq_len, vocab_size)
        logits = self.classifier(attn_output)
        return logits
