---
title: "[백테스팅] Walk-forward Validation으로 모델 검증하기"
date: 2026-06-13 10:10:00 +0900
slug: backtesting
categories:
  - 머신러닝
tags:
  - 백테스팅
  - 검증
  - Walk-forward
toc: true
toc_sticky: true
---

회귀부터 TFT까지 여러 모델을 만들어봤다. 그런데 이 모델들을 어떻게 검증해야 "실제로 쓸 만하다"고 믿을 수 있을까? 일반적인 K-Fold 교차검증을 시계열에 그대로 쓰면 안 되는 이유와, 올바른 검증 방법인 **Walk-forward Validation(백테스팅)**을 이번 글에서 정리한다.

## K-Fold가 시계열에서 왜 위험한가

K-Fold는 데이터를 무작위로 섞어 나누기 때문에, 시계열에 그대로 적용하면 **미래 데이터로 과거를 예측하는 학습**이 일어날 수 있다.

```
원본 시계열:  [1월][2월][3월][4월][5월][6월] ...

K-Fold (무작위 분할):
  Fold 1 학습: [1월][3월][5월][6월]   검증: [2월][4월]
              └─────────┬─────────┘
                  3월, 5월로 2월을 검증 → 미래 정보가 학습에 섞임 (Data Leakage)
```

2월을 예측하는데 3월, 5월 데이터로 학습한 모델을 쓰는 것은 실제 서비스 환경에서는 불가능한 상황이다 — 실전에서는 항상 "과거만 알고 미래를 모르는" 상태에서 예측해야 한다. 이런 미래 정보 유입을 **데이터 누수(Data Leakage)**라 한다.

## Walk-forward Validation의 원리

Walk-forward Validation은 항상 "학습 구간은 검증 구간보다 과거"라는 시간 순서를 지킨다. 학습 구간을 점점 확장(또는 일정 크기로 이동)하면서, 그 다음 시점만 검증하는 과정을 반복한다.

![Walk-forward Validation 구조](/assets/images/walk_forward_diagram.png)
*Step이 진행될수록 Train 구간(파란색)이 점점 늘어나고, 그 바로 다음 한 시점만 Test(주황색)로 사용한다. 항상 Test는 Train보다 미래 시점이다.*

## 구현 — Ridge 회귀로 직접 백테스팅

```python
import numpy as np
from sklearn.linear_model import Ridge

np.random.seed(0)
t = np.linspace(0, 8 * np.pi, 200)
data = np.sin(t) + np.random.normal(0, 0.2, 200)

def create_features(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 10
X, y = create_features(data, SEQ_LEN)

initial_train = 100
preds, actuals = [], []
for i in range(initial_train, len(X)):
    model = Ridge(alpha=1.0).fit(X[:i], y[:i])   # i 시점까지만 학습
    preds.append(model.predict(X[i:i+1])[0])      # 그 다음 한 시점만 예측
    actuals.append(y[i])

preds = np.array(preds)
actuals = np.array(actuals)
rmse = np.sqrt(np.mean((actuals - preds) ** 2))
mae = np.mean(np.abs(actuals - preds))
mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
```

매 반복마다 모델을 `i` 시점까지의 데이터로만 다시 학습시키고, 바로 다음 시점 하나만 예측한다. 이 과정을 전체 구간에 반복하면 "실전과 동일한 조건"에서의 예측 성능을 얻을 수 있다.

![백테스팅 결과 — 예측 vs 실제, 오차 분포](/assets/images/backtesting_results.png)
*위는 Walk-forward 방식으로 얻은 예측값과 실제값의 비교, 아래는 그 오차의 분포다. 오차가 0을 중심으로 좌우 대칭에 가까울수록 모델에 구조적 편향(bias)이 없다는 뜻이다.*

## 데이터 누수 방지 체크리스트

| 항목 | 잘못된 예 | 올바른 예 |
|------|-----------|-----------|
| 정규화(Scaler) | 전체 데이터로 `fit` 후 분할 | 학습 데이터로만 `fit`, 검증 데이터는 `transform`만 |
| 결측치 보간 | 전체 데이터 기준으로 보간 | 학습 시점까지의 데이터만 사용해 보간 |
| 피처 생성(이동평균 등) | 미래 시점 포함해서 계산 | 해당 시점 이전 데이터만 사용 |
| 검증 분할 | 무작위 분할(K-Fold) | 시간 순서를 지키는 분할(Walk-forward) |

```python
from sklearn.preprocessing import StandardScaler

# 잘못된 예 — 전체 데이터로 fit
# scaler = StandardScaler().fit(all_data)

# 올바른 예 — 학습 데이터로만 fit
scaler = StandardScaler().fit(X[:initial_train])
X_train_scaled = scaler.transform(X[:initial_train])
X_test_scaled = scaler.transform(X[initial_train:])
```

## 딥러닝 모델의 백테스팅 — Fixed-Window 방식

LSTM/TFT처럼 학습 비용이 큰 모델은 매 시점마다 처음부터 재학습하기 어렵다. 이런 경우 학습 구간을 고정된 크기로 유지하면서 일정 주기(예: 매월)로만 재학습하는 **Fixed-Window 백테스팅**을 절충안으로 쓴다. 매 시점 재학습 대비 정확도는 약간 떨어질 수 있지만, 실무에서 감당 가능한 계산 비용으로 시간 순서를 지킨 검증을 할 수 있다.

## 시리즈를 마치며

회귀분석의 기본 원리에서 시작해 정규화, 평가지표, 시계열 분해, ARIMA, RNN/LSTM, TFT를 거쳐 마지막으로 이 모델들을 제대로 검증하는 방법까지 정리했다.

| 단계 | 다룬 내용 |
|------|-----------|
| 회귀분석 | 선형회귀, OLS와 경사하강법, 다중회귀와 다중공선성, 정규화(Ridge/Lasso/ElasticNet), 평가지표 |
| 시계열 기초 | 시계열 분해, 정상성, ACF/PACF, ARIMA |
| 딥러닝 | RNN, LSTM(단변량/다변량), TFT |
| 검증 | Walk-forward Validation, 데이터 누수 방지 |

이 흐름에서 가장 중요했던 한 가지를 꼽으면, "더 복잡한 모델이 항상 더 좋은 모델은 아니다"라는 점이다. 정규화는 일부러 모델을 덜 정확하게 만들어 일반화 성능을 얻었고, ARIMA는 단순하지만 해석 가능했고, TFT는 강력하지만 검증을 제대로 하지 않으면 그 강력함이 오히려 과적합으로 이어질 수 있다. 모델 선택은 항상 데이터의 성격과 검증 결과를 함께 보고 판단해야 한다는 것이 이 시리즈 전체를 관통하는 결론이다.
