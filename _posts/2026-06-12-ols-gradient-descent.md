---
title: "[회귀분석 #2] OLS와 경사하강법으로 최적의 직선 찾기"
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
