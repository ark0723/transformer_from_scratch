import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tokenizer import Tokenizer
from dataloader import TextDataset, DataCollator
from models import Transformer


if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. Dataset & Tokenizer Preparation
    # ---------------------------------------------------------
    corpus = [
        "The cat sat on the mat.",
        "I love natural language processing.",
        "How are you doing today?",
        "She sells seashells by the seashore.",
        "Artificial intelligence is transforming the world.",
        "Can you believe it's already July?",
        "Let's meet at 5 p.m. in the cafe.",
        "He didn’t know what to say.",
        "The weather is nice and sunny.",
    ]
    # tokenizer
    tokenizer = Tokenizer()
    tokenizer.build_vocab(corpus)

    dataset = TextDataset(corpus)
    collator = DataCollator(tokenizer, max_length=10)

    dataloader = DataLoader(
        dataset, batch_size=10, shuffle=True, collate_fn=collator, num_workers=0
    )

    # ---------------------------------------------------------
    # 2. Model Initialization
    # ---------------------------------------------------------
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using Device: {device}")

    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        emb_dim=128,
        hidden_dim=512,
        n_heads=4,
        n_layers=2,
        bias=True,
        dropout=0.1,
    ).to(device)

    # ---------------------------------------------------------
    # 3. Optimizer & Loss Function Setting
    # ---------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # loss: cross entropy loss
    # ignore_index: <pad> token -> padding token not considered in loss calculation!!
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # ---------------------------------------------------------
    # 4. Train Loop
    # ---------------------------------------------------------
    epochs = 10
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_idx, batch_data in enumerate(dataloader):
            # (batch_size, seq_len)
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            # -------------------------------------------------
            # simple test: Next Token Prediction Task
            # Input: "The cat sat" (from start to second-to-last)
            # Target: "cat sat on" (from second to last)
            # -------------------------------------------------
            train_input = input_ids[:, :-1]
            train_target = input_ids[:, 1:]
            train_mask = attention_mask[:, :-1]

            # 1. model forward : (batch_size, seq_len, vocab_size)
            predictions = model(train_input, pad_mask=train_mask, mode="post-norm")

            # 2. loss calculation
            # CrossEntropyLoss takes (N, C) and (N) inputs, so we flatten the dimensions.
            # predictions.view(-1, vocab_size): (B * (L-1), Vocab_Size)
            # train_target.reshape(-1): (B * (L-1))
            loss = criterion(
                predictions.view(-1, tokenizer.vocab_size), train_target.reshape(-1)
            )

            # 3. backward and optimization
            optimizer.zero_grad()  # initialize gradient to 0 to avoid gradient accumulation from previous batch
            loss.backward()  # pytorch automatically accumulates gradients (grad += new_grad)
            optimizer.step()  # update model parameters

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    print("Training completed!")
