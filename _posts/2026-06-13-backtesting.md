---
title: "[백테스팅] Walk-forward Validation으로 모델 검증하기"
date: 2026-06-13 13:00:00 +0900
categories:
  - ML
tags:
  - 백테스팅
  - Walk-forward
  - 시계열
  - 모델검증
toc: true
toc_sticky: true
---

## 들어가며

모델을 만들었으면 실제로 얼마나 잘 작동하는지 검증해야 한다.
일반적인 머신러닝에서는 train/test를 랜덤으로 나누지만,
**시계열에서는 이 방법이 틀렸다.**
미래 데이터가 과거 학습에 섞이는 **데이터 누수(Data Leakage)**가 발생하기 때문이다.

---

## 1. 왜 일반 Cross Validation이 안 되는가

```
일반 K-Fold:
Fold 1: [1,2,4,5] train → [3] test       ← 3번 이후 데이터로 3번 이전 예측 (누수!)
Fold 2: [1,3,4,5] train → [2] test       ← 마찬가지 문제

시계열에서는 반드시 과거 → 미래 방향으로만 검증해야 한다.
```

---

## 2. Walk-forward Validation

**Walk-forward Validation**은 시간 순서를 유지하면서
학습 구간을 점진적으로 확장해가며 검증하는 방법이다.

```
Step 1: [1~6] train → [7] test
Step 2: [1~7] train → [8] test
Step 3: [1~8] train → [9] test
Step 4: [1~9] train → [10] test
```

각 스텝에서 모델을 새로 학습하고 다음 시점을 예측한다.
실제 운용 환경과 가장 유사한 검증 방법이다.

---

## 3. 구현

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

font_candidates = [
    'Noto Sans CJK JP', 'Noto Sans CJK KR',
    'NanumGothic', 'Malgun Gothic', 'AppleGothic', 'DejaVu Sans'
]
available_fonts = {f.name for f in fm.fontManager.ttflist}
selected_font = next((f for f in font_candidates if f in available_fonts), 'DejaVu Sans')
plt.rcParams['font.family'] = selected_font
plt.rcParams['axes.unicode_minus'] = False

# 데이터 생성
np.random.seed(42)
n = 200
t = np.linspace(0, 8 * np.pi, n)
data = np.sin(t) + np.random.normal(0, 0.2, n)

# 슬라이딩 윈도우로 피처 생성
def create_features(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 10
X, y = create_features(data, SEQ_LEN)

# Walk-forward Validation
initial_train_size = 100
predictions = []
actuals = []

for i in range(initial_train_size, len(X)):
    X_train = X[:i]
    y_train = y[:i]
    X_test = X[i:i+1]
    y_test = y[i:i+1]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    predictions.append(pred[0])
    actuals.append(y_test[0])

predictions = np.array(predictions)
actuals = np.array(actuals)

rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae = np.mean(np.abs(actuals - predictions))
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.2f}%")

# 시각화
plt.figure(figsize=(14, 5))
plt.plot(actuals, label='실제값', alpha=0.8)
plt.plot(predictions, label='예측값', alpha=0.8)
plt.title('Walk-forward Validation 결과')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 4. 데이터 누수 방지 체크리스트

시계열 모델에서 자주 발생하는 누수 패턴들이다.

| 누수 유형 | 잘못된 예 | 올바른 방법 |
|---|---|---|
| 정규화 누수 | 전체 데이터로 scaler fit | train 데이터로만 fit |
| 피처 누수 | 미래값으로 만든 피처 사용 | lag 피처는 과거값만 |
| 타겟 누수 | 타겟과 직결된 변수 포함 | 도메인 지식으로 제거 |
| 분할 누수 | 랜덤 split | 시간 순서대로 split |

```python
from sklearn.preprocessing import MinMaxScaler

# 잘못된 방법
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1))  # 전체 데이터로 fit ← 누수

# 올바른 방법
train_data = data[:split]
test_data = data[split:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))  # train만으로 fit
test_scaled = scaler.transform(test_data.reshape(-1, 1))        # transform만 적용
```

---

## 5. LSTM에 Walk-forward 적용

딥러닝 모델은 매 스텝마다 재학습하면 시간이 너무 오래 걸린다.
실용적인 방법은 **고정 윈도우(Fixed Window)**를 사용하는 것이다.

```
Fixed Window (윈도우 크기 = 100):
Step 1: [1~100] train → [101] test
Step 2: [2~101] train → [102] test
Step 3: [3~102] train → [103] test
```

학습 구간을 고정 크기로 유지하면서 슬라이딩하는 방식이다.
전체 재학습보다 빠르고, 최근 데이터에 더 집중할 수 있다.

---

## 정리

- 시계열에서는 랜덤 split 금지 — 반드시 시간 순서 유지
- Walk-forward Validation이 실제 운용과 가장 유사한 검증 방법
- 정규화, 피처 생성 모두 train 기준으로만 처리
- 딥러닝은 Fixed Window 방식으로 효율적으로 검증

---

## 시리즈를 마치며

이 시리즈는 **선형 회귀**에서 시작해 **백테스팅**까지 다뤘다.

| 단계 | 내용 |
|---|---|
| 회귀분석 | 선형 회귀, OLS, 정규화, 평가지표 |
| 시계열 기초 | 정상성, ACF/PACF, ARIMA |
| 딥러닝 | RNN → LSTM → TFT |
| 검증 | Walk-forward Validation, 누수 방지 |

각 단계가 이전 단계의 한계를 극복하며 발전해온 흐름이다.
앞으로는 실제 데이터로 각 모델을 직접 실험하고 비교하는 내용을 다룰 예정이다.
