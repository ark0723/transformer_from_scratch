import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(
        self, emb_dim: int, n_heads: int, bias: bool = True, dropout: float = 0.1
    ):
        super().__init__()

        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads."

        self.head_dim = emb_dim // n_heads
        self.n_heads = n_heads

        # w_q, w_k, w_v initialization
        self.w_q = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=bias)
        self.w_k = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=bias)
        self.w_v = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=bias)
        # final context vector
        self.w_o = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        causal: bool = False,
    ):
        """
        Args:
            x: embedded tensor (batch_size, seq_len, emb_dim)
            pad_mask: padding mask (batch_size, seq_len)
        Return:
            context vector (batch_size, seq_len, emb_dim)
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

        # 3. Attention Scores computation: shape (batch_size, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # attention masking
        # input mask shape: (batch_size, seq_len)
        # mask shape change required for broadcasting: (batch_size, 1, 1, seq_len)
        # why (batch_size, 1, 1, seq_len)? -> because we need to broadcast the mask with k(key) axis
        ####### applied attention masking results example #######
        ## table -> K1, K2, K3 (padding)
        ## Q1    -> score, score, -inf
        ## Q2    -> score, score, -inf
        ## Q3    -> score, score, -inf

        # if we make mask shape(batch_size, 1, seq_len, 1), the mask will be broadcasted with q(query) axis
        # and the mask will be applied to the wrong axis
        # example:
        # q1 -> K1, K2, K3
        # q2 -> K1, K2, K3
        # q3(padding) -> -inf, -inf, -inf

        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)

        if causal:
            # attention score shape: (batch_size, n_heads, seq_len, seq_len)
            # causal mask shape: (1, 1, seq_len, seq_len) for broadcasting with attention scores
            causal_mask = (
                torch.tril(torch.ones(seq_len, seq_len, device=x.device))
                .unsqueeze(0)
                .unsqueeze(1)
            )
            # logical_or: return True if either of the conditions is True
            # shape: (batch_size, 1, seq_len, seq_len)
            combined_mask = torch.logical_or(pad_mask == 0, causal_mask == 0)
        else:
            # shape: (batch_size, 1, 1, seq_len)
            combined_mask = pad_mask == 0
        # at the position where combined_mask is True, fill the attention scores with -inf
        attn_scores = attn_scores.masked_fill(combined_mask, float("-inf"))

        # apply softmax and dropout
        # shape (batch_size, n_heads, seq_len, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 4. context vector
        # shape (batch_size, n_heads, seq_len, head_dim)
        context_vector_by_heads = torch.matmul(attn_weights, v)
        # 5. concatenate context vectors by heads : (batch_size, seq_len, emb_dim)
        context_vector = (
            context_vector_by_heads.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.n_heads * self.head_dim)
        )
        context_vector = self.w_o(context_vector)
        return context_vector


class PointWiseFeedForward(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()

        self.linear1 = nn.Linear(in_features=emb_dim, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=emb_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : tensor after multi-head attention and layer normalization (batch_size, seq_len, emb_dim)
            Return:
                ffn_output: output of point wise feed-forward network (batch_size, seq_len, emb_dim)
        """

        ffn_output = self.linear1(x)
        ffn_output = self.relu(ffn_output)
        ffn_output = self.linear2(ffn_output)

        return ffn_output


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module to add positional information to the embedded tensor.
    Note: positional encoding is not a learnable parameter, so we don't need to compute gradients for it.
    """

    def __init__(self, emb_dim: int, max_length: int = 5000):
        super().__init__()
        pos_encode = torch.zeros(max_length, emb_dim)
        # position: (max_length, 1)
        position = torch.arange(0, max_length).unsqueeze(1)

        # 2i index for even index (0, 2, 4, ...)
        dim_idx = torch.arange(0, emb_dim, step=2)

        # div_term = 1.0 / 10000**(dim_idx/max_length)
        # generally,original positional encoding formula is not used, but we can use one line of below to calculate div_term to get the stable result
        # a^b = e^(b*log(a)) -> (1/10000)^(2i/emb_dim) = e^(2i/emb_dim * log(1/10000))
        div_term = torch.exp(dim_idx * -(math.log(10000.0) / emb_dim))
        # sin(position * div_term) : apply toeven index (0, 2, 4, ...)
        pos_encode[:, 0::2] = torch.sin(position * div_term)
        # cos(position * div_term) : apply to odd index (1, 3, 5, ...)
        pos_encode[:, 1::2] = torch.cos(position * div_term)
        # (max_length, emb_dim) -> (1, max_length, emb_dim) for broadcasting with (batch_size, seq_len, emb_dim)
        pos_encode = pos_encode.unsqueeze(0)

        # register buffer(): make it as a module parameter to be saved and loaded with the model
        # 모델 학습에는 필요없는 텐서들을 내부적으로 이용할 떄 적용,
        # register_buffer('name', tensor)를 적용하면 해당 텐서는 모델에 파라미터가 아닌 모델 버퍼에 등록되어
        # 모델에서 self.name으로 접근할 수 있으며 model.state_dict()에도 해당 텐서가 저장됨
        self.register_buffer("pos_encode", pos_encode)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: embedded tensor (batch_size, seq_len, emb_dim)
        Return:
            embedded tensor with positional encoding (batch_size, seq_len, emb_dim)
        """
        # sentence length can be varied
        seq_len = x.size(1)
        # add positional encoding to the embedded tensor
        embedded = x + self.pos_encode[:, :seq_len, :]
        return embedded


class Encoder(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        n_heads: int,
        bias: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            emb_dim=emb_dim,
            n_heads=n_heads,
            bias=bias,
            dropout=dropout,
            causal=False,
        )

        self.ffn = PointWiseFeedForward(emb_dim=emb_dim, hidden_dim=hidden_dim)
        # nn.LayerNorm: have two learnabe parameters -> self.norm_attn and self.norm_ffn need to be seperately specified
        # self.norm_attn: for attention  distribution / self.norm_ffn: for FFN distribution
        self.norm_attn = nn.LayerNorm(normalized_shape=emb_dim)
        self.norm_ffn = nn.LayerNorm(normalized_shape=emb_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: None | torch.Tensor = None,
        mode: str = "post-norm",
    ):
        """
        Args:
            x: embedded tensor (batch_size, seq_len, emb_dim)
            pad_mask: padding mask (batch_size, seq_len)
            mode: "pre-norm" or "post-norm"
        """

        mode = mode.lower()

        assert mode in [
            "pre-norm",
            "post-norm",
        ], f"Invalid mode: {mode}. Must be 'pre-norm' or 'post-norm'."

        if mode == "pre-norm":
            # layer normalization: stabilize the result and pass it to the next layer
            # -> maximize training stability even with deep layers
            # dropout: add random noise to the output of the sublayer
            # residual add: add the noise to the original input
            attn_output = self.attention(self.norm_attn(x), pad_mask=pad_mask)
            attn_output = self.dropout(attn_output)
            x = x + attn_output

            ffn_output = self.ffn(self.norm_ffn(x))
            ffn_output = self.dropout(ffn_output)
            x = x + ffn_output

        else:  # post-norm
            # dropout: add random noise to the output of the sublayer
            # residual add: add the noise to the original input
            # layer normalization: stabilize the result and pass it to the next layer

            attn_output = self.attention(x, pad_mask=pad_mask)
            # apply droput before residual connection and layer normalization
            attn_output = self.dropout(attn_output)
            # residual connection(input before multi-head attention + output of multi-head attention) & layer normalization
            x = self.norm_attn(x + attn_output)

            ffn_output = self.ffn(x)
            ffn_output = self.dropout(ffn_output)
            x = self.norm_ffn(x + ffn_output)

        return x


class Decoder(nn.Module):
    pass
