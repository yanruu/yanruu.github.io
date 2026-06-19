---
title: "[딥러닝 #2] LSTM 구조와 단변량 예측"
date: 2026-06-13 09:40:00 +0900
slug: lstm-univariate
categories:
  - 머신러닝
tags:
  - 딥러닝
  - LSTM
  - 시계열
toc: true
toc_sticky: true
---

지난 글에서 RNN이 먼 과거 정보를 기억하지 못하는 기울기 소실 문제를 확인했다. **LSTM(Long Short-Term Memory)**은 이 문제를 게이트(gate) 구조로 해결한 모델이다. 이번 글에서는 LSTM 셀 내부를 게이트별로 뜯어보고, 가장 간단한 형태인 단변량(univariate) 시계열 예측에 적용해본다.

## LSTM 셀 내부 구조

LSTM은 은닉 상태 $h_t$ 외에 **셀 상태(cell state) $C_t$** 를 별도로 유지한다. 셀 상태는 정보를 오래 보존하는 "장기 기억 저장소" 역할을 하고, 세 개의 게이트가 이 저장소에 무엇을 넣고 뺄지를 결정한다.

![LSTM 셀 내부 구조](/assets/images/lstm_cell_structure.png)
*셀 상태($C_{t-1} \to C_t$)가 맨 위 화살표를 따라 곧게 흐른다. 각 게이트가 이 흐름에 정보를 더하거나(Input), 지우거나(Forget), 내보낼지(Output) 결정한다.*

**Forget Gate** — 이전 셀 상태 중 얼마나 잊을지 결정.

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

**Input Gate** — 새 정보를 얼마나 받아들일지 결정.

$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \quad \tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$$

**Cell State 업데이트** — 잊을 건 잊고, 받아들일 건 더한다.

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output Gate** — 갱신된 셀 상태 중 얼마나 다음 은닉 상태로 내보낼지 결정.

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(C_t)$$

게이트들이 모두 $\sigma$(시그모이드, 0~1 범위)를 쓰기 때문에 "얼마나"라는 비율로 정보를 통제할 수 있고, 셀 상태가 곱셈이 아닌 덧셈으로 갱신되기 때문에 기울기가 RNN보다 훨씬 멀리까지 전파된다 — 지난 글의 기울기 소실 문제가 완화되는 이유다.

## PyTorch로 단변량 예측 구현

사인파(sine wave)처럼 패턴이 분명한 단변량 시계열로 LSTM을 학습시켜본다.

```python
import numpy as np
import torch
import torch.nn as nn

# 1. 데이터 생성
t = np.linspace(0, 8 * np.pi, 500)
data = np.sin(t)

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 20
X, y = create_sequences(data, SEQ_LEN)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

# 2. 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 3. 학습
for epoch in range(50):
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"epoch {epoch}, loss {loss.item():.4f}")

# 4. 평가
model.eval()
with torch.no_grad():
    pred_test = model(X_test_t).squeeze().numpy()
rmse = np.sqrt(np.mean((pred_test - y_test) ** 2))
print(f"Test RMSE: {rmse:.4f}")
```

학습이 끝나면 실제값과 예측값을 겹쳐 그려 패턴을 잘 따라가는지 시각적으로 확인하는 것이 좋다.

![LSTM 단변량 예측 결과](/assets/images/lstm_univariate_prediction.png)
*테스트 구간에서 LSTM 예측값(주황 점선)이 실제값(파란 실선)의 사인파 패턴을 거의 그대로 따라간다. 약간의 떨림은 학습 노이즈와 시퀀스 길이(20스텝) 제약에서 비롯된다.*

## 하이퍼파라미터가 결과에 미치는 영향

| 하이퍼파라미터 | 너무 작으면 | 너무 크면 |
|---------------|-------------|-----------|
| `hidden_size` | 패턴을 표현할 용량 부족 (과소적합) | 학습 느려지고 과적합 위험 |
| `SEQ_LEN` (시퀀스 길이) | 짧은 패턴만 학습 가능 | 학습 데이터 수 감소, 연산량 증가 |
| 학습률(`lr`) | 수렴이 느림 | 손실이 진동하거나 발산 |
| epoch 수 | 충분히 학습되지 않음 | 과적합 |

실제로는 이 표의 각 항목을 그리드서치나 직접 비교 실험으로 튜닝한다. 다음 글에서는 변수를 하나만 쓰는 단변량을 넘어, 여러 변수를 동시에 입력으로 쓰는 **다변량 LSTM**을 다룬다.

다음 글: **[다변량 LSTM — 여러 변수를 동시에 활용하기](/ml/lstm-multivariate/)**
