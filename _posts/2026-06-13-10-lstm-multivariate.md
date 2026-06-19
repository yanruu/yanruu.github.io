---
title: "[딥러닝 #3] 다변량 LSTM — 여러 변수를 동시에 활용하기"
date: 2026-06-13 09:50:00 +0900
slug: lstm-multivariate
categories:
  - ML
tags:
  - 딥러닝
  - LSTM
  - 다변량
toc: true
toc_sticky: true
---

지난 글의 LSTM은 입력이 하나의 변수(단변량)였다. 하지만 실무 데이터는 보통 예측 대상에 영향을 주는 여러 변수가 함께 존재한다. 예를 들어 기온을 예측할 때 습도, 풍속을 같이 참고하면 더 정확해질 수 있다. 이번 글에서는 **다변량(multivariate) LSTM**으로 이를 구현한다.

## 단변량과 다변량의 차이 — 입력 텐서 모양

단변량 LSTM의 입력은 `(batch, seq_len, 1)`이었다. 다변량은 마지막 차원이 변수 개수만큼 늘어난다 — `(batch, seq_len, n_features)`.

```python
import numpy as np
import pandas as pd

np.random.seed(1)
n = 500
t = np.linspace(0, 8 * np.pi, n)

feature1 = np.sin(t) + np.random.normal(0, 0.1, n)          # 주요 변수
feature2 = np.cos(t) + np.random.normal(0, 0.1, n)          # 보조 변수 1
feature3 = 0.5 * np.sin(2 * t) + np.random.normal(0, 0.1, n)  # 보조 변수 2
target = np.sin(t + 0.1)

df = pd.DataFrame({'feature1': feature1, 'feature2': feature2,
                    'feature3': feature3, 'target': target})
print(df.corr()['target'])
```

## 보조 변수가 정말 도움이 되는지 먼저 확인

다변량으로 만들기 전에, 추가하려는 변수들이 실제로 타깃과 관련이 있는지 상관관계로 먼저 점검하는 것이 좋다. 관련 없는 변수를 넣으면 모델이 오히려 노이즈를 학습해 성능이 떨어질 수 있다.

![변수 간 상관관계 히트맵](/assets/images/feature_correlation_multivariate.png)
*feature1은 target과 상관관계가 높아(0.9 이상) 포함할 가치가 분명하다. feature2, feature3도 일정 수준 연관되어 있어 다변량 입력으로 추가할 근거가 있다.*

## 시퀀스 생성과 모델 구조

```python
def create_multivariate_sequences(features, target, seq_len):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i+seq_len])
        y.append(target[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 20
features = df[['feature1', 'feature2', 'feature3']].values
X, y = create_multivariate_sequences(features, df['target'].values, SEQ_LEN)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
```

```python
import torch
import torch.nn as nn

class MultivariateLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = MultivariateLSTM(input_size=3)
```

`input_size=3`이 핵심 변경점이다. 모델 구조 자체(게이트 수식)는 단변량과 동일하지만, 각 게이트의 가중치 행렬 $W$가 3개 변수를 모두 받을 수 있도록 차원이 늘어난다.

![다변량 LSTM 구조](/assets/images/multivariate_lstm_architecture.png)
*세 개의 입력 변수가 하나의 시퀀스로 묶여 LSTM에 들어가고, LSTM은 이를 통합한 은닉 상태를 거쳐 하나의 예측값을 출력한다.*

## 학습과 평가는 단변량과 동일

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

for epoch in range(50):
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    optimizer.step()
```

학습 루프 자체는 단변량 코드와 거의 같다 — 차이는 입력 데이터의 마지막 차원뿐이다. 이것이 LSTM 구조의 장점이다: 변수가 늘어나도 모델의 핵심 로직은 그대로 유지된다.

## 다변량으로 확장할 때 주의할 점

| 항목 | 주의사항 |
|------|----------|
| 변수 선택 | 타깃과 관련 없는 변수를 넣으면 노이즈만 추가됨 — 상관관계/도메인 지식으로 사전 검토 |
| 스케일링 | 변수마다 단위/범위가 다르면 반드시 정규화(StandardScaler 등) 필요 |
| 결측치 | 한 변수라도 결측이 있으면 시퀀스 전체가 깨짐 — 전처리 단계에서 처리 필수 |
| 차원의 저주 | 변수가 너무 많아지면 학습 데이터량 대비 모델이 과적합되기 쉬움 |

다변량 LSTM은 변수를 단순히 "쌓는" 방식이라 모든 변수를 동등하게 취급한다. 다음 글에서 다룰 **TFT(Temporal Fusion Transformer)**는 여기서 한 단계 나아가 "어떤 변수가 지금 더 중요한지"까지 모델이 학습하게 만드는 구조다.

다음 글: **[TFT — Temporal Fusion Transformer](/ml/tft/)**
