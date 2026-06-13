---
title: "[딥러닝 #2] LSTM 구조와 단변량 예측"
date: 2026-06-13 12:30:00 +0900
categories:
  - ML
tags:
  - 딥러닝
  - LSTM
  - 시계열
  - 단변량
toc: true
toc_sticky: true
---

## 들어가며

[지난 글](../rnn)에서 RNN의 기울기 소실 문제를 확인했다.
**LSTM(Long Short-Term Memory)**은 게이트(Gate) 구조를 추가해
중요한 정보는 오래 기억하고, 불필요한 정보는 잊는 방식으로 이 문제를 해결한다.

---

## 1. LSTM의 핵심 — Cell State

RNN의 hidden state $h_t$ 하나로 모든 것을 처리했다면,
LSTM은 **Cell State $C_t$** 를 추가해 장기 기억을 별도로 관리한다.

- $C_t$ : 장기 기억 (컨베이어 벨트처럼 흘러가며 필요한 정보만 수정)
- $h_t$ : 단기 기억 (현재 시점의 출력)

---

## 2. LSTM의 세 가지 게이트

### Forget Gate — 무엇을 잊을까

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

- 0에 가까우면 이전 기억 삭제, 1에 가까우면 유지
- $\sigma$ : 시그모이드 함수 (출력 범위 0~1)

### Input Gate — 무엇을 기억할까

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

- $i_t$ : 새 정보를 얼마나 반영할지
- $\tilde{C}_t$ : 새로 추가할 후보 기억

### Cell State 업데이트

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

- 이전 기억을 얼마나 잊고, 새 기억을 얼마나 추가할지 결합

### Output Gate — 무엇을 출력할까

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(C_t)$$

---

## 3. 단변량 LSTM 예측 구현

입력 변수가 하나(예: 종가, 기온)인 단변량 시계열 예측이다.

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_candidates = [
    'Noto Sans CJK JP', 'Noto Sans CJK KR',
    'NanumGothic', 'Malgun Gothic', 'AppleGothic', 'DejaVu Sans'
]
available_fonts = {f.name for f in fm.fontManager.ttflist}
selected_font = next((f for f in font_candidates if f in available_fonts), 'DejaVu Sans')
plt.rcParams['font.family'] = selected_font
plt.rcParams['axes.unicode_minus'] = False

# 데이터 생성 (sin 곡선 + 노이즈)
np.random.seed(42)
t = np.linspace(0, 8 * np.pi, 500)
data = np.sin(t) + np.random.normal(0, 0.1, 500)

# 정규화
data_min, data_max = data.min(), data.max()
data_norm = (data - data_min) / (data_max - data_min)

# 슬라이딩 윈도우로 데이터셋 생성
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 20
X, y = create_sequences(data_norm, SEQ_LEN)

# train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Tensor 변환
X_train = torch.FloatTensor(X_train).unsqueeze(-1)
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_train = torch.FloatTensor(y_train).unsqueeze(-1)
y_test = torch.FloatTensor(y_test).unsqueeze(-1)

# LSTM 모델
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습
EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# 평가
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()
    y_true = y_test.numpy()

rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
print(f"RMSE: {rmse:.4f}")

# 시각화
plt.figure(figsize=(12, 4))
plt.plot(y_true, label='실제값')
plt.plot(y_pred, label='예측값')
plt.legend()
plt.title('LSTM 단변량 예측 결과')
plt.tight_layout()
plt.show()
```

---

## 4. 주요 하이퍼파라미터

| 파라미터 | 설명 | 일반적인 범위 |
|---|---|---|
| seq_len | 입력 시퀀스 길이 | 10~100 |
| hidden_size | LSTM 셀 크기 | 32~256 |
| num_layers | LSTM 레이어 수 | 1~3 |
| learning_rate | 학습률 | 0.001~0.01 |
| batch_size | 배치 크기 | 32~128 |

---

## 정리

- LSTM은 Cell State로 장기 기억을 별도 관리
- Forget / Input / Output 세 게이트로 정보 흐름 제어
- 단변량 예측: 슬라이딩 윈도우로 시퀀스 생성 후 학습
- 하이퍼파라미터 튜닝이 성능에 큰 영향을 줌

다음 글: **다변량 LSTM — 여러 변수를 동시에 활용하기**
