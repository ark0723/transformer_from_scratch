import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tokenizer import Tokenizer
from dataloader import TranslationDataset, Seq2SeqCollator
from models import Transformer

from torchinfo import summary

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. Dataset & Tokenizer Preparation (Source: KR, Target: EN)
    # ---------------------------------------------------------
    src_corpus = [
        "나는 학생 입니다",
        "고양이 가 매트 위에 있다",
        "오늘 날씨 가 좋다",
        "딥러닝 은 재미있다",
        "트랜스포머 모델 은 강력하다",
        "너 는 무엇 을 하고 있니",
    ]
    tgt_corpus = [
        "I am a student",
        "The cat is on the mat",
        "The weather is good today",
        "Deep learning is fun",
        "Transformer models are powerful",
        "What are you doing",
    ]

    # ---------------------------------------------------------
    # 2. Tokenizer Initialization & Build Vocab
    # ---------------------------------------------------------
    src_tokenizer = Tokenizer()
    src_tokenizer.build_vocab(src_corpus)  # Build Korean Vocab

    tgt_tokenizer = Tokenizer()
    tgt_tokenizer.build_vocab(tgt_corpus)

    print(f"Source Vocab Size (KR): {src_tokenizer.vocab_size}")
    print(f"Target Vocab Size (EN): {tgt_tokenizer.vocab_size}")
    # check special token IDs
    print(
        f"Target SOS ID: {tgt_tokenizer.sos_token_id}, EOS ID: {tgt_tokenizer.eos_token_id}"
    )

    # ---------------------------------------------------------
    # 3. Dataset & DataLoader
    # ---------------------------------------------------------
    # return raw string pairs
    dataset = TranslationDataset(src_corpus, tgt_corpus)

    # Collator: Tokenize and special token injection
    collator = Seq2SeqCollator(src_tokenizer, tgt_tokenizer, max_length=12)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collator)

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using Device: {device}")

    model = Transformer(
        src_vocab_size=src_tokenizer.vocab_size,
        target_vocab_size=tgt_tokenizer.vocab_size,
        emb_dim=128,
        hidden_dim=512,
        n_heads=4,
        n_layers=2,
        bias=True,
        dropout=0.1,
    ).to(device)

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # loss: cross entropy loss
    # ignore_index: <pad> token -> padding token not considered in loss calculation!!
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_token_id)

    epochs = 30
    model.train()
    # 3. summary 함수 호출
    # print("\n--- Model Summary (Actual Data) ---")
    # summary(
    #     model,
    #     # tensor input is passed in tuple format to input_data
    #     input_data=(train_input, train_mask),
    #     kwargs={"mode": "post-norm"},
    #     col_names=["input_size", "output_size", "num_params"],
    #     row_settings=["var_names"],
    #     verbose=1,
    #     depth=5,
    # )
    # print("-----------------------------------\n")

    print("\n--- Start Training ---")
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in dataloader:
            # Collator returned dictionary keys are used to extract data
            src_ids = batch["src_ids"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_input_ids = batch["tgt_input_ids"].to(device)  # <sos> ...
            tgt_mask = batch["tgt_mask"].to(device)
            tgt_label_ids = batch["tgt_label_ids"].to(device)  # ... <eos>

            # Forward Pass
            # - src_mask: Encoder의 Padding을 가리기 위해 사용 (Cross Attention)
            # - tgt_mask: Decoder의 Padding을 가리기 위해 사용 (Self Attention)

            predictions = model(
                encoder_input=src_ids,
                decoder_input=tgt_input_ids,
                encoder_pad_mask=src_mask,
                decoder_pad_mask=tgt_mask,
                mode="post-norm",
            )

            # Loss calculation
            # CrossEntropyLoss takes (N, C) and (N) inputs, so we flatten the dimensions.
            # predictions.view(-1, vocab_size): (B * (L-1), Vocab_Size)
            # train_target.reshape(-1): (B * (L-1))
            loss = criterion(
                predictions.view(-1, tgt_tokenizer.vocab_size), tgt_label_ids.view(-1)
            )

            # backpropagation
            optimizer.zero_grad()  # initialize gradient to 0 to avoid gradient accumulation from previous batch
            loss.backward()  # pytorch automatically accumulates gradients (grad += new_grad)
            optimizer.step()  # update model parameters

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    print("Training completed!")

    # ---------------------------------------------------------
    # 6. Inference Test (Greedy Decoding)
    # ---------------------------------------------------------
    print("\n--- Inference Test ---")
    model.eval()

    test_sentence = "나는 학생 입니다"
    print(f"Input: {test_sentence}")

    # 1. Encode Source (수동 처리)
    # Tokenizer의 __call__은 배치를 가정하므로, 단일 문장 처리를 위해 리스트로 감쌈
    # max_length는 넉넉하게
    src_encoded = src_tokenizer([test_sentence], max_length=12)
    src_tensor = torch.tensor(src_encoded["input_ids"], dtype=torch.long).to(device)
    src_mask = torch.tensor(src_encoded["attention_mask"], dtype=torch.long).to(device)

    # 2. Prepare Decoder Input
    # Start with <sos> (Tokenizer에 저장된 ID 사용)
    decoder_input = torch.tensor([[tgt_tokenizer.sos_token_id]], dtype=torch.long).to(
        device
    )

    # 3. Generation Loop
    with torch.no_grad():
        for _ in range(12):  # Max generation length
            # 현재 decoder_input 길이에 맞춰 mask 생성 (Padding 없음)
            tgt_mask = (decoder_input != tgt_tokenizer.pad_token_id).long().to(device)

            # Forward
            logits = model(src_tensor, decoder_input, src_mask, tgt_mask)

            # 다음 토큰 예측 (마지막 시점의 Logits)
            next_token_id = logits[:, -1, :].argmax(dim=-1).item()

            # <eos>가 나오면 종료
            if next_token_id == tgt_tokenizer.eos_token_id:
                break

            # 입력에 추가
            decoder_input = torch.cat(
                [decoder_input, torch.tensor([[next_token_id]], device=device)], dim=1
            )

    # 4. Decode Result
    # decoder_input: [[<sos>, token1, token2, ...]]
    # <sos>는 제외하고 디코딩
    generated_ids = decoder_input.squeeze().cpu().numpy()[1:]

    # [NEW] Tokenizer의 개선된 decode 메서드 활용 (skip_special_tokens=True)
    # ndarray가 반환되므로 ' '.join()으로 문자열 변환
    decoded_tokens = tgt_tokenizer.decode(generated_ids, skip_special_tokens=True)
    translated_text = " ".join(decoded_tokens)

    print(f"Translated: {translated_text}")
