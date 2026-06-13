---
title: "[딥러닝 #3] 다변량 LSTM — 여러 변수를 동시에 활용하기"
date: 2026-06-13 12:40:00 +0900
categories:
  - ML
tags:
  - 딥러닝
  - LSTM
  - 시계열
  - 다변량
toc: true
toc_sticky: true
---

## 들어가며

[지난 글](../lstm-univariate)에서는 입력 변수가 하나인 단변량 LSTM을 다뤘다.
현실 데이터는 여러 변수가 동시에 영향을 준다.
예를 들어 전력 수요 예측에는 기온, 습도, 요일, 시간대 등이 함께 작용한다.
**다변량 LSTM**은 이런 여러 변수를 동시에 입력으로 받아 예측한다.

---

## 1. 단변량 vs 다변량

| | 단변량 | 다변량 |
|---|---|---|
| 입력 변수 | 1개 | N개 |
| input_size | 1 | N |
| 데이터 형태 | (batch, seq_len, 1) | (batch, seq_len, N) |
| 활용 예시 | 종가만으로 예측 | 종가 + 거래량 + 기술지표 |

구조는 단변량과 동일하고, **input_size만 변수 개수로 바꾸면 된다.**

---

## 2. 다변량 LSTM 구현

```python
import numpy as np
import torch
import torch.nn as nn
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

# 다변량 데이터 생성 (3개 변수)
np.random.seed(42)
n = 500
t = np.linspace(0, 8 * np.pi, n)

feature1 = np.sin(t) + np.random.normal(0, 0.1, n)        # 주요 변수
feature2 = np.cos(t) + np.random.normal(0, 0.1, n)        # 보조 변수 1
feature3 = 0.5 * np.sin(2*t) + np.random.normal(0, 0.1, n) # 보조 변수 2
target = np.sin(t + 0.1)                                   # 예측 대상

# 정규화
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

features = np.stack([
    normalize(feature1),
    normalize(feature2),
    normalize(feature3)
], axis=1)  # shape: (500, 3)
target_norm = normalize(target)

# 슬라이딩 윈도우
def create_sequences(features, target, seq_len):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i+seq_len])
        y.append(target[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 20
X, y = create_sequences(features, target_norm, SEQ_LEN)

split = int(len(X) * 0.8)
X_train = torch.FloatTensor(X[:split])
X_test = torch.FloatTensor(X[split:])
y_train = torch.FloatTensor(y[:split]).unsqueeze(-1)
y_test = torch.FloatTensor(y[split:]).unsqueeze(-1)

# 다변량 LSTM 모델 — input_size=3으로 변경
class MultivariateLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = MultivariateLSTM()
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
plt.title('다변량 LSTM 예측 결과')
plt.tight_layout()
plt.show()
```

---

## 3. 변수 선택이 중요한 이유

변수가 많다고 무조건 좋지 않다.
관련 없는 변수가 많으면 오히려 노이즈가 늘어 성능이 떨어질 수 있다.

**변수 선택 방법:**

- **상관관계 분석**: target과 상관계수가 높은 변수 선택
- **Feature Importance**: 트리 기반 모델로 중요도 확인 후 선택
- **도메인 지식**: 실제로 영향을 주는 변수인지 판단

```python
import pandas as pd

df = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'feature3': feature3,
    'target': target
})

print(df.corr()['target'].sort_values(ascending=False))
```

---

## 4. 다변량 LSTM의 한계

| 한계 | 설명 |
|---|---|
| 변수 간 관계 해석 어려움 | 어떤 변수가 얼마나 기여했는지 알기 어려움 |
| 긴 시퀀스에서 여전히 약함 | 장기 의존성 문제 완전 해결 아님 |
| 변수 중요도 반영 불가 | 모든 변수를 동일하게 처리 |

이 한계를 **Attention 메커니즘**과 **TFT**가 해결한다.

---

## 정리

- 다변량 LSTM은 input_size만 변수 개수로 바꾸면 구현 가능
- 데이터 형태: (batch, seq_len, 변수 개수)
- 변수 선택이 성능에 큰 영향을 줌
- 변수 중요도 해석 한계 → TFT로 이어짐

다음 글: **TFT — Temporal Fusion Transformer**
