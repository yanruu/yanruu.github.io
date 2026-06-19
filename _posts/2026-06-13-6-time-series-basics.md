---
title: "[시계열 #1] 시계열 데이터 기초 — 정상성, ACF/PACF"
date: 2026-06-13 09:10:00 +0900
slug: time-series-basics
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

지금까지는 행과 행 사이에 순서가 없는 데이터(회귀)를 다뤘다. 이번 글부터는 **시간 순서 자체가 정보인** 시계열(time series) 데이터를 다룬다. 시계열을 다루기 전에 반드시 알아야 할 세 가지 — 시계열 분해, 정상성, ACF/PACF를 정리한다.

## 시계열 데이터의 구성 요소

시계열은 보통 추세(Trend), 계절성(Seasonality), 잔차(Residual) 세 요소가 섞여 있다고 가정한다.

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
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

t = np.arange(120)
data = 0.05 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, 120)
ts = pd.Series(data, index=pd.date_range('2015-01', periods=120, freq='ME'))

result = seasonal_decompose(ts, model='additive')
result.plot()
plt.show()
```

![시계열 분해 — 추세, 계절성, 잔차](/assets/images/time_series_decomposition.png)
*맨 위가 원본 시계열, 그 아래 순서대로 추세, 계절성(12개월 주기), 그리고 이 셋으로 설명되지 않는 잔차다. 잔차에 패턴이 남아있지 않을수록 분해가 잘 된 것이다.*

이렇게 분해해보면 "이 시계열이 왜 이런 모양인지"가 구성 요소별로 분리되어, 이후 모델링 방향을 정하는 데 도움이 된다.

## 정상성(Stationarity)이란

대부분의 전통적 시계열 모델(ARIMA 등)은 데이터가 **정상(stationary)** 이라고 가정한다. 정상 시계열은 시간이 지나도 평균과 분산이 일정하고, 특정 시점 간의 자기상관 구조가 시간에 따라 변하지 않는다.

반대로 추세나 계절성이 있는 시계열은 비정상(non-stationary)이다. 위 그래프의 원본 시계열처럼 평균이 시간에 따라 증가하는 경우가 대표적이다.

정상성 여부는 ADF(Augmented Dickey-Fuller) 검정으로 통계적으로 확인할 수 있다.

```python
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(ts)
print(f"ADF 통계량: {adf_result[0]:.3f}")
print(f"p-value: {adf_result[1]:.3f}")
# p-value < 0.05 이면 "정상이다"라는 대안가설을 채택 (귀무가설: 비정상)
```

이 예시 데이터는 추세(0.05*t)가 섞여 있어서 ADF 검정의 p-value가 0.05보다 크게 나온다 — 즉 비정상으로 판정된다. 다음 글에서 다룰 ARIMA는 이 비정상성을 차분(differencing)으로 제거한 뒤 모델을 적용한다.

## ACF / PACF — 자기상관을 보는 두 가지 창

ACF(자기상관함수, Autocorrelation Function)는 시점 $t$와 $t-k$ 사이의 상관관계를, PACF(편자기상관함수, Partial Autocorrelation Function)는 중간 시점들의 영향을 제거한 뒤 $t$와 $t-k$의 순수한 상관관계를 보여준다.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ts, lags=30, ax=axes[0])
plot_pacf(ts, lags=30, ax=axes[1], method='ywm')
plt.show()
```

![ACF와 PACF 비교](/assets/images/acf_pacf_example.png)
*ACF(왼쪽)는 12개월 주기로 값이 반복해서 튀는 모습을 보여 계절성을 드러낸다. PACF(오른쪽)는 직접적인 시차 관계만 남겨서 어느 시차까지 진짜 의존성이 있는지를 보여준다.*

이 두 그래프는 다음 글의 ARIMA에서 AR항과 MA항의 차수(p, q)를 정하는 데 직접 쓰인다 — PACF가 빠르게 0 근처로 줄어드는 시차가 AR 차수의 힌트, ACF가 빠르게 줄어드는 시차가 MA 차수의 힌트다.

## 정리

| 개념 | 의미 | 확인 방법 |
|------|------|-----------|
| 추세/계절성/잔차 분해 | 시계열을 구성 요소로 분리 | `seasonal_decompose` |
| 정상성 | 평균·분산이 시간에 따라 불변 | ADF 검정 |
| ACF | 시차별 전체 상관관계 | `plot_acf` |
| PACF | 시차별 순수 상관관계 | `plot_pacf` |

이 네 가지 개념이 시계열 분석의 출발점이다. 다음 글에서는 정상성을 확보한 뒤 실제로 예측 모델을 만드는 첫 단계, ARIMA를 다룬다.

다음 글: **[ARIMA — 시계열 예측의 고전 모델](/ml/arima/)**
