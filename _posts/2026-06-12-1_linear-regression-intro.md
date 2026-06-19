---
title: "[회귀분석 #1] 선형 회귀란 무엇인가"
date: 2026-06-12 09:00:00 +0900
slug: linear-regression-intro
categories:
  - ML
tags:
  - 회귀분석
  - 선형회귀
  - 머신러닝
toc: true
toc_sticky: true
---

## 왜 회귀분석부터 시작하는가

딥러닝, LSTM, Transformer 같은 복잡한 모델들도 결국 "입력값으로 출력값을 예측한다"는 본질은 같다.
그 본질을 가장 단순하게 표현한 것이 **선형 회귀(Linear Regression)**다.

---

## 선형 회귀란

### 선형과 비선형

**선형(Linear)**이란 입력과 출력의 관계가 **직선**으로 표현되는 것을 말한다.
반대로 **비선형(Non-linear)**은 곡선, 지수, 사인 곡선처럼 직선으로 표현할 수 없는 관계다.

예를 들어:
- 공부 시간이 늘수록 시험 점수가 일정하게 올라간다 → **선형**
- 연습량이 늘수록 처음엔 빠르게, 나중엔 더디게 실력이 오른다 → **비선형**

선형 회귀는 이 중 **선형 관계를 가정하고 데이터에 직선을 맞추는 모델**이다.

### 직선을 맞춘다는 것

공부 시간(x)과 시험 점수(y)의 관계를 알고 싶다고 하자.
데이터를 점으로 찍으면 대략 우상향하는 패턴이 보인다.
선형 회귀는 이 점들을 가장 잘 설명하는 직선 하나를 찾는 것이다.

![산점도와 회귀선](/assets/images/linear_regression_plot1.png)

---

## 직선의 방정식

그 직선을 수식으로 표현하면:

$$y = wx + b$$

- $w$ : 기울기 (가중치, weight) — 입력이 1 증가할 때 출력이 얼마나 변하는지
- $b$ : 절편 (편향, bias) — 입력이 0일 때의 출력값
- $x$ : 입력값
- $y$ : 예측값

$w$와 $b$ 값에 따라 직선의 모양이 달라진다.
아래 그래프에서 각 경우를 확인할 수 있다:

![w, b 변화에 따른 직선](/assets/images/linear_regression_plot2.png)

- **w가 클수록** 직선이 가파르게 올라간다
- **b가 클수록** 직선이 위로 이동한다

---

## 어떻게 최적의 직선을 찾는가

모든 점과의 거리 오차를 최소화하는 직선을 찾으면 된다.
이때 사용하는 것이 **MSE(평균 제곱 오차)**다:

$$MSE = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

- $y_i$ : 실제값
- $\hat{y}_i$ : 예측값
- $n$ : 데이터 개수

MSE를 최소화하는 $w$와 $b$를 구하는 방법이 **OLS(최소제곱법)** 또는 **경사하강법(Gradient Descent)**이다.
다음 포스팅에서 이 두 가지를 자세히 다룬다.

> RMSE, MAE, MAPE 등 다른 평가지표가 궁금하다면 → [[회귀분석 #5] 모델 성능 평가](/ml/evaluation-metrics/)

---

## 직접 실험해보기

VS Code에서 `.ipynb` 파일로 직접 실행해볼 수 있다.
아래 코드를 새 노트북 셀에 붙여넣고 실행해보자.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 데이터 생성
np.random.seed(42)
x = np.linspace(1, 10, 30)
y = 5 * x + 20 + np.random.normal(0, 8, 30)

# 모델 학습
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

print(f"w (기울기): {model.coef_[0]:.2f}")
print(f"b (절편): {model.intercept_:.2f}")

# 시각화
y_pred = model.predict(x.reshape(-1, 1))
plt.scatter(x, y, color='steelblue', label='데이터')
plt.plot(x, y_pred, color='tomato', label='회귀선')
plt.xlabel('공부 시간 (h)')
plt.ylabel('시험 점수')
plt.legend()
plt.show()
```

w와 b 값을 직접 바꿔가며 직선이 어떻게 변하는지 확인해보면
수식의 의미를 훨씬 빠르게 체감할 수 있다.

---

## 정리

| 개념 | 설명 |
|------|------|
| 선형 회귀 | 데이터에 직선을 맞추는 예측 모델 |
| 가중치(w) | 직선의 기울기, 입력의 영향력 |
| 편향(b) | 직선의 절편 |
| MSE | 예측 오차를 수치화한 손실 함수 |

다음 글: **[OLS와 경사하강법으로 최적의 직선 찾기](/ml/ols-gradient-descent/)**
