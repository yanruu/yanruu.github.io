---
title: "[딥러닝 #1] RNN 구조 이해 — 순환 신경망의 등장"
date: 2026-06-13 12:20:00 +0900
categories:
  - ML
tags:
  - 딥러닝
  - RNN
  - 시계열
  - 순환신경망
toc: true
toc_sticky: true
---

## 들어가며

[지난 글](../arima)에서 ARIMA의 한계를 확인했다.
선형 관계만 모델링하고, 장기 의존성을 잘 포착하지 못한다.
이를 극복하기 위해 등장한 것이 **RNN(Recurrent Neural Network)**이다.

---

## 1. 일반 신경망의 한계

일반 **MLP(Multi-Layer Perceptron)**는 입력을 독립적으로 처리한다.
시계열처럼 **순서가 중요한 데이터**에서는 이전 시점의 정보를 기억하지 못한다.

예를 들어 "오늘 기온은 어제보다 높다"를 예측하려면 어제 기온을 알아야 하는데,
MLP는 각 입력을 독립적으로 보기 때문에 이런 맥락을 반영하기 어렵다.

---

## 2. RNN 구조

RNN은 **이전 시점의 출력(hidden state)을 다음 시점의 입력으로 재사용**한다.

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

$$\hat{y}_t = W_y h_t + b_y$$

- $h_t$ : 현재 시점의 hidden state (기억)
- $h_{t-1}$ : 이전 시점의 hidden state
- $x_t$ : 현재 시점의 입력
- $W_h, W_x, W_y$ : 학습되는 가중치

핵심은 **같은 가중치를 모든 시점에서 공유**한다는 점이다.

---

## 3. RNN 구현

```python
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

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

model = SimpleRNN()
print(model)
```

---

## 4. RNN의 한계 — 기울기 소실

RNN은 역전파(BPTT) 과정에서 시점이 길어질수록 기울기가 점점 작아지는 문제가 발생한다.

$$\frac{\partial L}{\partial W} = \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

- 시퀀스가 길면 곱셈이 반복되어 기울기가 0에 수렴
- 결과적으로 **먼 과거의 정보를 잊어버림**

이 문제를 해결하기 위해 등장한 것이 **LSTM**이다.

---

## 5. RNN vs LSTM 비교

| | RNN | LSTM |
|---|---|---|
| 구조 | 단순 순환 | 게이트 구조 추가 |
| 장기 의존성 | 약함 | 강함 |
| 기울기 소실 | 발생 | 완화 |
| 계산 비용 | 낮음 | 높음 |

---

## 정리

- RNN은 이전 시점의 hidden state를 재사용해 순서 정보를 반영
- 같은 가중치를 모든 시점에서 공유
- 시퀀스가 길어지면 기울기 소실 문제 발생
- 이를 해결한 것이 LSTM

다음 글: **LSTM 구조와 단변량 예측**
