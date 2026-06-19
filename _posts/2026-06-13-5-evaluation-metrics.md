---
title: "[회귀분석 #5] 모델 성능 평가 — R², RMSE, MAE, MAPE"
date: 2026-06-13 09:00:00 +0900
slug: evaluation-metrics
categories:
  - ML
tags:
  - 회귀분석
  - 평가지표
  - RMSE
  - MAE
  - R2
toc: true
toc_sticky: true
---

지난 글에서 Ridge/Lasso로 과적합을 줄였다. 그런데 "줄였다"는 걸 어떻게 숫자로 증명할까? 회귀 모델 성능을 비교할 때 가장 많이 쓰는 네 가지 지표 — MSE/RMSE, MAE, MAPE, R² — 를 수식, 코드, 그리고 직접 그려본 그래프로 정리한다.

## MSE / RMSE — 오차를 제곱해서 본다

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 \qquad RMSE = \sqrt{MSE}$$

오차를 제곱하기 때문에 **큰 오차에 더 큰 페널티**를 준다. RMSE는 MSE에 제곱근을 씌워 원래 데이터와 같은 단위로 돌려준 값이라 해석이 더 직관적이다.

```python
import numpy as np

y_true = np.array([3, 5, 4, 7, 6, 5, 4, 6])
y_pred = np.array([2.8, 5.2, 3.9, 7.5, 5.8, 5.1, 4.2, 5.9])

mse = np.mean((y_true - y_pred) ** 2)
rmse = np.sqrt(mse)
print(f"MSE: {mse:.3f}, RMSE: {rmse:.3f}")
```

## MAE — 오차의 절댓값 평균

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

제곱하지 않으므로 모든 오차를 동등하게 다룬다. 이상치(outlier)에 RMSE보다 덜 민감하다.

```python
mae = np.mean(np.abs(y_true - y_pred))
print(f"MAE: {mae:.3f}")
```

## RMSE는 왜 이상치에 민감한가 — 직접 확인

한 데이터 포인트의 오차만 점점 키워보면 RMSE와 MAE가 어떻게 다르게 반응하는지 바로 보인다.

```python
outlier_shift = np.linspace(0, 20, 25)
rmse_vals, mae_vals = [], []
for s in outlier_shift:
    yp2 = y_pred.copy()
    yp2[0] = y_pred[0] + s          # 한 점만 점점 빗나가게 만듦
    rmse_vals.append(np.sqrt(np.mean((y_true - yp2) ** 2)))
    mae_vals.append(np.mean(np.abs(y_true - yp2)))
```

![이상치 하나가 커질 때 RMSE와 MAE의 반응 차이](/assets/images/metrics_outlier_sensitivity.png)
*한 점의 오차만 키웠을 때 RMSE는 제곱 때문에 가속도가 붙듯 커지지만, MAE는 선형으로만 증가한다.*

이상치가 실제로 의미 있는 신호(예: 드문 고가 매물)라면 RMSE가 그 신호를 더 강하게 반영해 적합하고, 이상치가 단순 노이즈/오류라면 MAE가 모델 평가를 덜 왜곡시켜 더 안정적이다.

## MAPE — 퍼센트로 본 오차

$$MAPE = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| \%$$

단위와 무관하게 "평균적으로 몇 % 빗나갔는가"를 알려줘서 서로 다른 스케일의 모델/데이터를 비교할 때 유용하다. 단, $y_i$가 0에 가까우면 값이 폭발하므로 매출·인구처럼 0에 가까운 값이 거의 없는 데이터에 적합하다.

```python
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(f"MAPE: {mape:.2f}%")
```

## R² — 설명력

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

분모는 "평균만으로 예측했을 때의 오차", 분자는 "모델로 예측했을 때의 오차"다. R²가 1에 가까울수록 모델이 데이터의 분산을 잘 설명한다는 뜻이고, 0이면 평균으로 찍는 것과 다를 게 없다는 뜻이다. 음수가 나올 수도 있는데, 이는 모델이 평균보다도 못한 예측을 하고 있다는 신호다.

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")
```

## 잔차 플롯으로 한눈에 진단하기

지표 하나의 숫자만 보고는 모델이 "왜" 안 좋은지 알기 어렵다. 잔차(실제값 - 예측값)를 예측값에 대해 그려보면 모델이 놓치고 있는 패턴이 바로 드러난다.

![좋은 모델과 나쁜 모델의 잔차 플롯 비교](/assets/images/residual_plot_example.png)
*좋은 모델은 잔차가 0을 중심으로 무작위로 흩어진다. 나쁜 모델은 잔차에 곡선 형태의 패턴이 남아있다 — 이는 모델이 포착하지 못한 비선형 관계가 남아있다는 뜻이다.*

잔차에 어떤 패턴(곡선, 깔때기 모양 등)이 보인다면, 그건 모델 구조 자체를 다시 봐야 한다는 신호다. 단순히 지표 숫자만 좋다고 끝이 아니라는 게 이 그래프의 핵심이다.

## 어떤 지표를 써야 할까

| 지표 | 이상치 민감도 | 단위 | 적합한 상황 |
|------|--------------|------|-------------|
| RMSE | 높음 | 원래 단위 | 큰 오차를 특히 피해야 할 때 (예: 가격 예측) |
| MAE | 낮음 | 원래 단위 | 이상치에 흔들리지 않는 평가가 필요할 때 |
| MAPE | 중간 | % | 스케일이 다른 데이터/모델 비교 |
| R² | - | 0~1 (비율) | 모델의 전반적 설명력 요약 |

실무에서는 한 지표만 보지 않고 RMSE + R²처럼 묶어서 함께 확인하는 경우가 많다. 다음 글부터는 회귀를 떠나 시계열 데이터로 넘어간다. 정상성, ACF/PACF 같은 시계열 고유의 개념부터 시작한다.

다음 글: **[시계열 데이터 기초 — 정상성, ACF/PACF](/ml/time-series-basics/)**
