# import torch

# print(f"PyTorch version: {torch.__version__}")
# print(f"MPS available: {torch.backends.mps.is_available()}")
# print(f"MPS built: {torch.backends.mps.is_built()}")

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     x = torch.randn(1000, 1000, device=device)
#     y = torch.randn(1000, 1000, device=device)
#     z = torch.matmul(x, y)
#     print("✅ MPS 가속 성공!")

import torch
import transformers
from transformers import AutoModel, AutoTokenizer
import numpy as np

print("=== 고급 호환성 테스트 ===")

# Transformers + PyTorch + MPS 통합 테스트
try:
    print("🔄 BERT 모델 로딩 중...")
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # MPS 디바이스로 모델 이동
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        print("✅ BERT 모델을 MPS 디바이스로 이동 성공")

        # 간단한 추론 테스트
        text = "Hello, this is a test sentence."
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        print("✅ MPS에서 BERT 추론 성공")
        print(f"   출력 텐서 shape: {outputs.last_hidden_state.shape}")
        print(f"   출력 텐서 device: {outputs.last_hidden_state.device}")
    else:
        print("⚠️ MPS 사용 불가, CPU에서 테스트")

except Exception as e:
    print(f"❌ Transformers + MPS 테스트 실패: {str(e)}")

# PyTorch와 NumPy 상호 호환성 테스트
try:
    print("\\n🔄 PyTorch-NumPy 상호 호환성 테스트...")

    # NumPy -> PyTorch
    np_array = np.random.randn(5, 5)
    torch_tensor = torch.from_numpy(np_array)

    # PyTorch -> NumPy
    torch_result = torch.matmul(torch_tensor, torch_tensor.T)
    np_result = torch_result.numpy()

    print("✅ PyTorch-NumPy 상호 변환 성공")

except Exception as e:
    print(f"❌ PyTorch-NumPy 호환성 테스트 실패: {str(e)}")

print("\\n=== 메모리 사용량 체크 ===")
if torch.backends.mps.is_available():
    # 큰 텐서로 메모리 테스트
    try:
        device = torch.device("mps")
        large_tensor = torch.randn(1000, 1000, device=device)
        result = torch.matmul(large_tensor, large_tensor.T)
        print("✅ 대용량 텐서 MPS 연산 성공")
        print(f"   텐서 크기: {large_tensor.shape}")

        # 메모리 정리
        del large_tensor, result
        torch.mps.empty_cache()
        print("✅ MPS 캐시 정리 성공")

    except Exception as e:
        print(f"⚠️ 대용량 텐서 테스트 실패: {str(e)}")

print("\\n🎯 전체 호환성 검사 완료!")
