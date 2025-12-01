import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Module.

    This module handles both **Self-Attention** and **Cross-Attention** mechanisms.

    ### Understanding Query, Key, and Value (Q, K, V):
    The core of attention is mapping a Query against a set of Keys to compute scores,
    which are then used to weight the Values.

    1. **Self-Attention (Encoder & Decoder):**
       - **Source:** The input tensor `x` itself.
       - **Mechanism:** $Q, K, V$ are all derived from `x`.
       - **Concept:** "How much does this token (Query) relate to other tokens in the same sentence (Keys)?"

    2. **Cross-Attention (Decoder-only):**
       - **Source:** - **Query ($Q$):** Comes from the Decoder's input `x` (the target sentence generated so far).
         - **Key ($K$) & Value ($V$):** Come from the Encoder's output `context` (the full source sentence).
       - **Mechanism:** $Q$ comes from `x`, while $K, V$ come from `context`.
       - **Concept:** "How much does the current token I'm generating (Query) relate to the input source sentence tokens (Keys)?"

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        n_heads (int): Number of attention heads.
        bias (bool): Whether to add bias to the linear projections.
        dropout (float): Dropout probability.
        is_cross_attn (bool): If True, performs Cross-Attention. If False, performs Self-Attention.
        is_causal (bool): If True, applies a causal mask (masked self-attention) to prevent attending to future tokens.
        max_seq_len (int): Maximum sequence length for pre-registering the causal mask buffer.
    """

    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        bias: bool = True,
        dropout: float = 0.1,
        is_cross_attn: bool = False,
        is_causal: bool = False,
        max_seq_len: int = 5000,
    ):
        super().__init__()

        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads."

        self.head_dim = emb_dim // n_heads
        self.n_heads = n_heads
        self.is_cross_attn = is_cross_attn
        self.is_causal = is_causal
        self.max_seq_len = max_seq_len

        # Initialization of projection layers for Q, K, V
        self.w_q = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=bias)
        self.w_k = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=bias)
        self.w_v = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=bias)

        # final context vector
        self.w_o = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

        # Register causal mask buffer only if it is Self-Attention and Causal mode is enabled.
        # This improves memory efficiency by avoiding re-creation of the mask at every forward pass.
        if not is_cross_attn and is_causal:
            # shape: (1, 1, max_seq_len, max_seq_len)
            causal_mask = (
                torch.tril(torch.ones(max_seq_len, max_seq_len))
                .unsqueeze(0)
                .unsqueeze(1)
            )
            # register_buffer saves this tensor to the state_dict but does not update it during backprop.
            self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        key_pad_mask: torch.Tensor | None = None,
    ):
        """
        Forward pass for Multi-Head Attention.

        Args:
            x (torch.Tensor): Input tensor acting as the Query source.
                Shape: (batch_size, seq_len_q, emb_dim)
            context (torch.Tensor | None): Context tensor acting as Key/Value source (from Encoder).
                Required only for Cross-Attention.
                Shape: (batch_size, seq_len_k, emb_dim)
            key_pad_mask (torch.Tensor | None): Mask to ignore padding tokens in Keys.
                Shape: (batch_size, seq_len_k)
                Values: 1 for valid tokens, 0 for padding (to be masked).

        Returns:
            torch.Tensor: Context vector (output of attention).
                Shape: (batch_size, seq_len_q, emb_dim)
        """
        batch_size, seq_len_q, emb_dim = x.size()

        # ----------------------------------------------------------------------
        # 1. Compute Q, K, V
        # ----------------------------------------------------------------------

        # Query (Q) always comes from 'x'.
        # In Decoder: 'x' is the target sequence.
        # In Encoder: 'x' is the source sequence.
        q = self.w_q(x)

        # Determine source for Key (K) and Value (V)
        if self.is_cross_attn:
            assert (
                self.is_causal == False
            ), "Causal attention is not allowed in cross attention."
            assert (
                context is not None
            ), "Context (Encoder output) is required for Cross-Attention."

            # Cross-Attention: K and V come from the Encoder output (context).
            k = self.w_k(context)
            v = self.w_v(context)
            # Cross Attn에서는 seq_len이 source(encoder)의 길이임
            # 따라서 split 할 때 view의 차원을 context 의 seq_len_k 기준으로 맞춰야 함
            seq_len_k = context.size(1)
        else:
            # Self-Attention: K and V come from 'x' (same as Q).
            k = self.w_k(x)
            v = self.w_v(x)
            seq_len_k = seq_len_q

        # ----------------------------------------------------------------------
        # 2. Split Heads & Reshape (for Multi-Head Attention)
        # ----------------------------------------------------------------------
        # Shape: (batch_size, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.n_heads, self.head_dim).transpose(1, 2)

        # ----------------------------------------------------------------------
        # 3. Compute Attention Scores
        # ----------------------------------------------------------------------
        # Shape: (batch_size, n_heads, seq_len_q, seq_len_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ----------------------------------------------------------------------
        # 4. Apply Masking
        # ----------------------------------------------------------------------

        # A. Causal Masking (Look-ahead Mask)
        # Applied only for Decoder Self-Attention (is_causal=True).
        if self.is_causal:
            # Retrieve pre-calculated mask from buffer and slice it to current sequence length.
            if seq_len_q > self.max_seq_len:
                mask = (
                    torch.tril(torch.ones(seq_len_q, seq_len_q, device=x.device))
                    .unsqueeze(0)
                    .unsqueeze(1)
                )
            else:
                mask = self.causal_mask[:, :, :seq_len_q, :seq_len_q]

            # Mask future tokens with -inf (where mask is 0)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # B. Padding Masking (Key Padding Mask)
        # Why (batch_size, 1, 1, seq_len_k)?
        # We need to broadcast the mask along the Query axis (dim 2) and Head axis (dim 1).
        # The mask effectively says: "For ANY Query token, do not attend to THESE Key tokens (padding)."
        #
        # [Example Broadcasting]
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

        if key_pad_mask is not None:
            assert key_pad_mask.dim() == 2, "key_pad_mask must be a 2D tensor."

            # Validation: Ensure mask length matches the Key/Value length.
            if self.is_cross_attn:
                assert (
                    key_pad_mask.size(1) == seq_len_k
                ), f"Cross Attention: key_pad_mask's sequence length, ({key_pad_mask.size(1)}) must match context vector's sequence length ({seq_len_k})"
            else:
                assert (
                    key_pad_mask.size(1) == seq_len_q
                ), f"Self Attention: key_pad_mask's sequence length, ({key_pad_mask.size(1)}) must match query vector's sequence length ({seq_len_q})"

            # Expand for broadcasting: (batch_size, seq_len_k) -> (batch_size, 1, 1, seq_len_k)
            mask_expanded = key_pad_mask.unsqueeze(1).unsqueeze(2)
            # Apply mask: 0 indicates padding (ignore), so fill with -inf
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, float("-inf"))

        # ----------------------------------------------------------------------
        # 5. Output Computation
        # ----------------------------------------------------------------------

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of Values (V)
        # Shape: (batch_size, n_heads, seq_len_q, head_dim) -> (batch_size, seq_len_q, emb_dim)
        context_vector_by_heads = torch.matmul(attn_weights, v)

        # Concatenate heads and restore original embedding dimension
        # Shape: (batch_size, seq_len_q, emb_dim)
        context_vector = (
            context_vector_by_heads.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, self.n_heads * self.head_dim)
        )

        # Final linearprojection
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


class EncoderLayer(nn.Module):
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
            is_cross_attn=False,
            is_causal=False,
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
        key_pad_mask: None | torch.Tensor = None,
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
            attn_output = self.attention(self.norm_attn(x), key_pad_mask=key_pad_mask)
            attn_output = self.dropout(attn_output)
            x = x + attn_output

            ffn_output = self.ffn(self.norm_ffn(x))
            ffn_output = self.dropout(ffn_output)
            x = x + ffn_output

        else:  # post-norm
            # dropout: add random noise to the output of the sublayer
            # residual add: add the noise to the original input
            # layer normalization: stabilize the result and pass it to the next layer

            attn_output = self.attention(x, key_pad_mask=key_pad_mask)
            # apply droput before residual connection and layer normalization
            attn_output = self.dropout(attn_output)
            # residual connection(input before multi-head attention + output of multi-head attention) & layer normalization
            x = self.norm_attn(x + attn_output)

            ffn_output = self.ffn(x)
            ffn_output = self.dropout(ffn_output)
            x = self.norm_ffn(x + ffn_output)

        return x


class DecoderLayer(nn.Module):
    """
    Single Decoder layer for the Transformer model.

    Logic flow: Masked Self-Attention -> Cross-Attention -> Point Wise Feed Forward
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        n_heads: int,
        bias: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1. Masked Self-Attention (Causal)
        # "지금까지 생성한 단어들끼리 어떤 연관성이 있는가?"를 계산
        self.causal_attn = MultiHeadAttention(
            emb_dim=emb_dim,
            n_heads=n_heads,
            bias=bias,
            dropout=dropout,
            is_cross_attn=False,
            is_causal=True,  # 중요: Decoder Self-Attention은 미래를 못 봄
        )

        # 2. Cross-Attention
        # "내가 지금 번역(생성)하려는 단어가 원문(Source)의 어느 부분과 연관되어 있는가?"
        self.cross_attn = MultiHeadAttention(
            emb_dim=emb_dim,
            n_heads=n_heads,
            bias=bias,
            dropout=dropout,
            is_cross_attn=True,  # 중요: Encoder Output을 참조
            is_causal=False,
            max_seq_len=5000,
        )

        self.ffn = PointWiseFeedForward(emb_dim=emb_dim, hidden_dim=hidden_dim)
        # nn.LayerNorm: have two learnabe parameters -> self.norm_attn, self.norm_cross_attn, and self.norm_ffn need to be seperately specified
        # self.norm_attn: for attention  distribution / self.norm_cross_attn: for cross attention distribution / self.norm_ffn: for FFN distribution
        self.norm_attn = nn.LayerNorm(normalized_shape=emb_dim)
        self.norm_cross_attn = nn.LayerNorm(normalized_shape=emb_dim)
        self.norm_ffn = nn.LayerNorm(normalized_shape=emb_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        source_pad_mask: None | torch.Tensor = None,
        target_pad_mask: None | torch.Tensor = None,
        mode: str = "post-norm",
    ):
        """
        Args:
            x: Decoder input (batch_size, seq_len_tgt, emb_dim)
            encoder_output: Encoder output context (batch_size, seq_len_src, emb_dim)
            source_pad_mask: Padding mask for Encoder (src sequence) -> used in Cross-Attn
            target_pad_mask: Padding mask for Decoder (tgt sequence) -> used in Self-Attn
            mode: "pre-norm" or "post-norm"
        """

        mode = mode.lower()

        assert mode in [
            "pre-norm",
            "post-norm",
        ], f"Invalid mode: {mode}. Must be 'pre-norm' or 'post-norm'."

        if mode == "pre-norm":
            # Masked Self-Attention: Norm -> Attn -> Residual Add
            # target_pad_mask 사용 (디코더 입력의 패딩 처리)
            attn_output = self.causal_attn(
                self.norm_attn(x), key_pad_mask=target_pad_mask
            )
            attn_output = self.dropout(attn_output)
            x = x + attn_output

            # Cross-Attention: Norm -> Attn -> Residual Add
            # source_pad_mask 사용 (인코더 출력의 패딩 처리)
            attn_output = self.cross_attn(
                self.norm_cross_attn(x),
                context=encoder_output,
                key_pad_mask=source_pad_mask,
            )
            attn_output = self.dropout(attn_output)
            x = x + attn_output

            # Point Wise Feed Forward: Norm -> FFN -> Residual Add
            ffn_output = self.ffn(self.norm_ffn(x))
            ffn_output = self.dropout(ffn_output)
            x = x + ffn_output

        else:  # post-norm
            # Masked Self-Attention
            attn_output = self.causal_attn(x, key_pad_mask=target_pad_mask)
            attn_output = self.dropout(attn_output)
            x = self.norm_attn(x + attn_output)
            # Cross-Attention
            attn_output = self.cross_attn(
                x, context=encoder_output, key_pad_mask=source_pad_mask
            )
            attn_output = self.dropout(attn_output)
            x = self.norm_cross_attn(x + attn_output)

            # Point Wise Feed Forward
            ffn_output = self.ffn(x)
            ffn_output = self.dropout(ffn_output)
            x = self.norm_ffn(x + ffn_output)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
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
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_encoding = PositionalEncoding(emb_dim, max_length=max_length)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(emb_dim, hidden_dim, n_heads, bias, dropout)
                for _ in range(n_layers)
            ]
        )
        # pre-norm mode only : final layer normalization
        self.norm = nn.LayerNorm(normalized_shape=emb_dim)

    def forward(
        self, x: torch.Tensor, key_pad_mask: torch.Tensor, mode: str = "post-norm"
    ):
        """
        Args:
            x: input ids (batch_size, seq_len)
            pad_mask: padding mask (batch_size, seq_len)
        """
        x = self.embedding(x) * math.sqrt(self.emb_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, key_pad_mask=key_pad_mask, mode=mode)

        if mode == "pre-norm":
            x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
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
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_encoding = PositionalEncoding(emb_dim, max_length=max_length)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(emb_dim, hidden_dim, n_heads, bias, dropout)
                for _ in range(n_layers)
            ]
        )
        # pre-norm mode only : final layer normalization
        self.norm = nn.LayerNorm(normalized_shape=emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        source_pad_mask: torch.Tensor,
        target_pad_mask: torch.Tensor,
        mode: str = "post-norm",
    ):
        """
        Args:
            x: input ids (batch_size, seq_len)
            pad_mask: padding mask (batch_size, seq_len)
        """
        x = self.embedding(x) * math.sqrt(self.emb_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output, source_pad_mask, target_pad_mask, mode=mode)

        if mode == "pre-norm":
            x = self.norm(x)

        return x
