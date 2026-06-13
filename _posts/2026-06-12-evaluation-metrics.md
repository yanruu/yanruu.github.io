---
title: "[회귀분석 #5] 모델 성능 평가 — R², RMSE, MAE, MAPE"
date: 2026-06-13 12:00:00 +0900
categories:
  - ML
tags:
  - 회귀분석
  - 평가지표
  - RMSE
  - MAE
  - R²
toc: true
toc_sticky: true
---

## 들어가며

모델을 만들었으면 얼마나 잘 예측하는지 수치로 확인해야 한다.
회귀 모델에서 자주 쓰이는 평가지표 네 가지를 정리한다.

---

## 1. MSE / RMSE

**MSE(Mean Squared Error)**: 오차의 제곱 평균

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**RMSE(Root MSE)**: MSE에 루트를 씌워 단위를 원래대로 복원

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

- 오차가 클수록 제곱으로 인해 크게 패널티를 줌
- 이상치에 민감
- **단위가 y와 같아서 해석이 직관적**

```python
import numpy as np

y_true = np.array([3, 5, 4, 7, 6])
y_pred = np.array([2.8, 5.2, 3.9, 7.5, 5.8])

mse = np.mean((y_true - y_pred) ** 2)
rmse = np.sqrt(mse)
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}")
```

---

## 2. MAE

**MAE(Mean Absolute Error)**: 오차의 절댓값 평균

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

- 이상치에 덜 민감 (제곱 없음)
- RMSE보다 해석이 단순
- "평균적으로 얼마나 틀렸는가"를 직접 표현

```python
mae = np.mean(np.abs(y_true - y_pred))
print(f"MAE: {mae:.4f}")
```

---

## 3. MAPE

**MAPE(Mean Absolute Percentage Error)**: 오차를 실제값 대비 비율로 표현

$$MAPE = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100$$

- **단위에 무관**하게 비교 가능 (%)
- 실제값이 0에 가까우면 값이 폭발적으로 커지는 단점
- 시계열 예측에서 자주 사용

```python
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(f"MAPE: {mape:.2f}%")
```

---

## 4. R² (결정계수)

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- 모델이 데이터 분산을 얼마나 설명하는지 나타냄
- 범위: $-\infty$ ~ 1 (1에 가까울수록 좋음)
- 0이면 평균으로만 예측하는 것과 같은 수준
- 음수면 평균보다도 못한 모델

```python
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"R²: {r2:.4f}")
```

---

## 5. 지표 비교 및 선택 기준

| 지표 | 이상치 민감도 | 단위 | 주 사용처 |
|---|---|---|---|
| RMSE | 높음 | y와 같음 | 일반 회귀 |
| MAE | 낮음 | y와 같음 | 이상치 많을 때 |
| MAPE | 낮음 | % | 시계열, 비교 |
| R² | 높음 | 없음 (0~1) | 모델 설명력 확인 |

**시계열 예측**에서는 RMSE + MAPE를 함께 보는 경우가 많다.
RMSE로 절대적인 오차 크기를, MAPE로 상대적인 오차 비율을 확인한다.

---

## 정리

- **RMSE**: 오차 크기, 이상치에 민감, 단위 직관적
- **MAE**: 평균 오차, 이상치에 강건
- **MAPE**: 비율 오차, 단위 무관, y=0이면 사용 불가
- **R²**: 모델 설명력, 1에 가까울수록 좋음

다음 글: **시계열 데이터 기초 — 정상성, ACF/PACF**
