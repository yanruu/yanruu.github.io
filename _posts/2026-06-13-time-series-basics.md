---
title: "[시계열 #1] 시계열 데이터 기초 — 정상성, ACF/PACF"
date: 2026-06-13 12:10:00 +0900
categories:
  - ML
tags:
  - 시계열
  - 정상성
  - ACF
  - PACF
toc: true
toc_sticky: true
---

## 들어가며

회귀분석은 변수 간의 관계를 모델링했다.
시계열은 **시간의 흐름에 따른 데이터의 패턴**을 모델링한다.
주가, 기온, 수요량처럼 시간 순서가 중요한 데이터가 여기 해당한다.

LSTM, TFT 같은 딥러닝 모델도 결국 시계열 데이터를 다루는 것이므로
기초 개념을 탄탄히 잡고 가는 것이 중요하다.

---

## 1. 시계열 데이터의 구성 요소

시계열 데이터는 보통 네 가지 요소로 분해된다:

- **추세(Trend)**: 장기적으로 증가하거나 감소하는 방향성
- **계절성(Seasonality)**: 일정 주기로 반복되는 패턴 (월별, 요일별 등)
- **주기(Cycle)**: 불규칙한 장기 파동
- **잔차(Residual)**: 위 세 요소로 설명되지 않는 노이즈

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.font_manager as fm

font_candidates = [
    'Noto Sans CJK JP', 'Noto Sans CJK KR',
    'NanumGothic', 'Malgun Gothic', 'AppleGothic', 'DejaVu Sans'
]
available_fonts = {f.name for f in fm.fontManager.ttflist}
selected_font = next((f for f in font_candidates if f in available_fonts), 'DejaVu Sans')
plt.rcParams['font.family'] = selected_font
plt.rcParams['axes.unicode_minus'] = False

# 예시 데이터 생성
import numpy as np
np.random.seed(42)
t = np.arange(120)
data = 0.05 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, 120)
ts = pd.Series(data, index=pd.date_range('2015-01', periods=120, freq='ME'))

result = seasonal_decompose(ts, model='additive')
result.plot()
plt.tight_layout()
plt.show()
```

---

## 2. 정상성 (Stationarity)

**정상성**이란 시계열의 통계적 특성(평균, 분산)이 시간에 따라 변하지 않는 성질이다.

- **정상 시계열**: 평균과 분산이 일정, 자기상관이 시차에만 의존
- **비정상 시계열**: 추세나 계절성이 있어 평균/분산이 변함

대부분의 통계 모델(ARIMA 등)은 정상 시계열을 가정한다.
비정상이면 **차분(differencing)**으로 정상화한다.

$$y'_t = y_t - y_{t-1}$$

---

## 3. 정상성 검정 — ADF 테스트

**ADF(Augmented Dickey-Fuller) 검정**으로 정상성을 통계적으로 확인한다.

- 귀무가설: 단위근이 있다 (비정상)
- p-value < 0.05 → 귀무가설 기각 → **정상**

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(ts)
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")

if result[1] < 0.05:
    print("정상 시계열")
else:
    print("비정상 시계열 → 차분 필요")
```

---

## 4. ACF (자기상관함수)

**ACF(AutoCorrelation Function)**는 시계열과 자기 자신의 과거값 사이의 상관관계를 시차별로 나타낸다.

$$ACF(k) = \text{Corr}(y_t, y_{t-k})$$

- lag k에서의 ACF가 크다 → k 시점 전 값이 현재에 영향을 줌
- **MA(q) 모델의 차수 결정**에 사용

```python
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(ts, lags=30)
plt.title("ACF")
plt.show()
```

---

## 5. PACF (편자기상관함수)

**PACF(Partial ACF)**는 중간 시차의 영향을 제거한 순수한 자기상관이다.

- **AR(p) 모델의 차수 결정**에 사용
- lag k 이후로 급격히 0에 수렴하면 AR(k) 모델이 적합

```python
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(ts, lags=30)
plt.title("PACF")
plt.show()
```

---

## 6. ACF vs PACF로 모델 결정

| 패턴 | 적합한 모델 |
|---|---|
| ACF 급격히 감소, PACF p 시차 후 절단 | AR(p) |
| ACF q 시차 후 절단, PACF 급격히 감소 | MA(q) |
| 둘 다 점진적으로 감소 | ARMA(p,q) |

---

## 정리

- 시계열은 추세, 계절성, 주기, 잔차로 구성된다
- 정상성: 평균/분산이 시간에 따라 일정한 성질
- ADF 검정으로 정상성 확인, 비정상이면 차분
- ACF → MA 차수, PACF → AR 차수 결정에 활용

다음 글: **ARIMA — 시계열 예측의 고전 모델**
