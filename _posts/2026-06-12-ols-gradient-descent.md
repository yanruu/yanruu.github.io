---
title: "[회귀분석 #2] OLS와 경사하강법으로 최적의 직선 찾기"
date: 2026-06-12 09:10:00
categories:
  - ML
tags:
  - 회귀분석
  - OLS
  - 경사하강법
  - 머신러닝
toc: true
toc_sticky: true
---

## 들어가며

[지난 글](../linear-regression-intro)에서 선형 회귀의 목표는 MSE를 최소화하는 $w$와 $b$를 찾는 것이라고 했다.
그 방법이 두 가지다. **OLS**는 수식으로 직접 구하고, **경사하강법**은 반복적으로 찾아간다.

---

## 1. OLS (최소제곱법)

OLS(Ordinary Least Squares)는 MSE를 미분해서 0이 되는 지점을 수식으로 직접 구한다.

단순 선형 회귀 $y = wx + b$ 에서 최적의 $w$, $b$는:

$$w = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

$$b = \bar{y} - w\bar{x}$$

- $\bar{x}$ : $x$의 평균
- $\bar{y}$ : $y$의 평균

수식 한 번으로 정확한 답이 나온다는 게 OLS의 장점이다.

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

x_mean, y_mean = x.mean(), y.mean()

w = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b = y_mean - w * x_mean

print(f"w = {w:.4f}, b = {b:.4f}")
# w = 0.6000, b = 1.8000
```

---

## 2. 경사하강법 (Gradient Descent)

경사하강법은 산에서 가장 가파른 방향으로 내려가듯, MSE가 줄어드는 방향으로 $w$와 $b$를 조금씩 업데이트한다.

업데이트 규칙:

$$w \leftarrow w - \alpha \frac{\partial MSE}{\partial w}$$

$$b \leftarrow b - \alpha \frac{\partial MSE}{\partial b}$$

- $\alpha$ : 학습률 (learning rate) — 한 번에 얼마나 이동할지
- 편미분 값이 기울기(gradient), 이 방향 반대로 이동

편미분을 전개하면:

$$\frac{\partial MSE}{\partial w} = -\frac{2}{n}\sum_{i=1}^{n}x_i(y_i - \hat{y}_i)$$

$$\frac{\partial MSE}{\partial b} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)$$

```python
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

w, b = 0.0, 0.0
lr = 0.01
n = len(x)

for epoch in range(1000):
    y_pred = w * x + b
    dw = -2/n * np.sum(x * (y - y_pred))
    db = -2/n * np.sum(y - y_pred)
    w -= lr * dw
    b -= lr * db

print(f"w = {w:.4f}, b = {b:.4f}")
# w ≈ 0.6000, b ≈ 1.8000
```

OLS와 같은 결과에 수렴하는 것을 확인할 수 있다.

---

## 3. OLS vs 경사하강법

| | OLS | 경사하강법 |
|---|---|---|
| 방식 | 수식으로 직접 계산 | 반복적으로 업데이트 |
| 속도 | 빠름 (1회 계산) | 느림 (수백~수천 반복) |
| 데이터 크기 | 대용량에 불리 (행렬 연산 비용) | 대용량에 유리 |
| 적용 범위 | 선형 모델만 가능 | 비선형, 딥러닝까지 확장 가능 |
| 하이퍼파라미터 | 없음 | 학습률($\alpha$) 설정 필요 |

**결론:** 데이터가 작고 선형 모델이면 OLS, 데이터가 크거나 딥러닝으로 확장할 거면 경사하강법.

---

## 정리

- OLS는 수식으로 단번에 최적해를 구한다
- 경사하강법은 기울기를 따라 반복적으로 최솟값을 찾아간다
- 두 방법 모두 같은 목표(MSE 최소화)를 향한다
- 딥러닝(LSTM, TFT 등)은 경사하강법의 확장선 위에 있다

다음 글: **다중 선형 회귀와 행렬 표현**
