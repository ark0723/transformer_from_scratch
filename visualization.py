import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from layers import PositionalEncoding
from tokenizer import Tokenizer
from models import Transformer


def show_positional_encoding(emb_dim: int, max_length: int):
    positional_encoding_layer = PositionalEncoding(emb_dim, max_length)
    # (1, max_length, emb_dim) -> (max_length, emb_dim) and move to cpu to convert to numpy
    pos_encode = positional_encoding_layer.pos_encode.squeeze().cpu().numpy()

    # plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    pos = ax.imshow(
        pos_encode, cmap="RdBu", extent=(0, pos_encode.shape[1], pos_encode.shape[0], 0)
    )
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Position in Sequence")
    ax.set_title("Positional Encoding over Embedding Dimensions")
    ax.set_xticks([i * 10 for i in range(0, pos_encode.shape[1] // 10)])
    ax.set_yticks([i * 10 for i in range(0, pos_encode.shape[0] // 10)])
    plt.show()


def show_tsne_embedding(
    model: Transformer, tokenizer: Tokenizer, example_text: str, max_length: int
):
    model.eval()
    # 1. tokenizing
    # tokenizer(text) -> {'input_ids': np.array, 'attention_mask': np.array}
    # input_ids, attention_mask -> (batch_size, max_length)
    inputs_dict = tokenizer(example_text, max_length=max_length)

    token_ids = inputs_dict["input_ids"][0]  # (max_length,)
    attention_mask = inputs_dict["attention_mask"][0]  # (max_length,)

    # 2. Filtering Padding
    valid_indices = token_ids[attention_mask == 1]
    print(
        f"Original length: {len(token_ids)}, Valid token length: {len(valid_indices)}"
    )

    # 2. get embeddings from model
    device = next(model.parameters()).device

    tensor_indices = torch.tensor(valid_indices, dtype=torch.long).to(device)
    # pass through embedding layer only for valid indices
    # model.embedding_layer(indices) -> (num_valid_tokens, emb_dim)
    with torch.no_grad():
        embedding_vectors = model.embedding_layer(tensor_indices).cpu().numpy()

    # decode token ids to tokens
    words = tokenizer.decode(valid_indices)

    # 3. apply TSNE
    if len(embedding_vectors) < 2:
        print("Not enough tokens to perform t-SNE (need at least 2 tokens).")
        return

    # perplexity should be less than the number of sample
    n_samples = len(embedding_vectors)
    perp = min(5, n_samples - 1)

    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    reduced_embeddings = tsne.fit_transform(embedding_vectors)

    # 4. plot
    plt.figure(figsize=(8, 6))
    for i, word in enumerate(words):
        x, y = reduced_embeddings[i]
        plt.scatter(x, y)
        plt.annotate(word, xy=(x + 0.1, y + 0.1), fontsize=12)
    plt.title("TSNE Embedding of Example Text")
    plt.grid(True)
    plt.show()


def show_attention_weights_matrix(
    model: Transformer, tokenizer: Tokenizer, example_text: str, max_length: int = 50
):
    model.eval()

    # 1. Prepare Input
    encoded_dict = tokenizer(example_text, max_length=max_length)
    input_ids = encoded_dict["input_ids"]  # (1, seq_len) or (batch, seq_len)
    attention_mask = encoded_dict["attention_mask"]  # (1, seq_len)

    device = next(model.parameters()).device
    input_tensor = torch.tensor(input_ids).to(device)
    mask_tensor = torch.tensor(attention_mask).to(device)

    # 입력이 1차원이면 배치 차원 추가
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)

    # 2. Forward Pass (Return 값 2개 받기)
    with torch.no_grad():
        logits, attn_weights = model(input_tensor, mask=mask_tensor)

    # attn_weights shape: (batch_size, n_heads, seq_len, seq_len)

    # 3. Process for Visualization
    # 3-1. 배치에서 꺼내기 (첫 번째 샘플)
    # shape: (n_heads, seq_len, seq_len)
    attn_map = attn_weights[0].cpu().numpy()

    # 3-2. 패딩(Padding) 부분 잘라내기 (시각화 깔끔하게 하기 위해)
    # 실제 유효한 토큰 길이 계산
    valid_len = int(mask_tensor[0].sum().item())

    # 유효한 부분만 슬라이싱: (n_heads, valid_len, valid_len)
    attn_map = attn_map[:, :valid_len, :valid_len]

    # 3-3. Head 처리 (선택: 평균 내기 or 특정 헤드 보기)
    # 여기서는 "모든 헤드의 평균"을 시각화합니다. (가장 일반적)
    # shape: (valid_len, valid_len)
    avg_attn_map = np.mean(attn_map, axis=0)

    # 4. Get Tokens for Labels
    # 유효한 토큰 ID만 가져와서 디코딩
    valid_ids = input_tensor[0][:valid_len].cpu().numpy()
    tokens = tokenizer.decode(valid_ids)

    # 5. Plotting (Seaborn Heatmap)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        avg_attn_map,
        annot=True,
        fmt=".2f",
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="gray",
        square=True,
        cbar=True,
    )

    plt.title(f"Self-Attention Weights (Avg of {model.n_heads} Heads)", fontsize=14)
    ax.set_xlabel("Key (Source)", labelpad=10, fontsize=12)
    ax.set_ylabel("Query (Target)", fontsize=12)

    # X축 라벨 위로 옮기기 (선택사항)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    plt.xticks(rotation=45, ha="left")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_positional_encoding(128, 50)
