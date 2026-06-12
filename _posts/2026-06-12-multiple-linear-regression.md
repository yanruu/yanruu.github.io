---
title: "[회귀분석 #3] 다중 선형 회귀와 행렬 표현"
categories:
  - ML
tags:
  - 회귀분석
  - 다중회귀
  - 선형대수
  - 머신러닝
toc: true
toc_sticky: true
---

## 들어가며

[지난 글](../ols-gradient-descent)에서는 입력 변수가 하나인 단순 선형 회귀를 다뤘다.
현실 데이터는 대부분 여러 변수가 동시에 영향을 미친다.
예를 들어 집값은 면적, 방 개수, 위치, 연식 등 여러 요소가 복합적으로 작용한다.
이처럼 입력 변수가 여러 개인 경우가 **다중 선형 회귀(Multiple Linear Regression)**다.

---

## 1. 다중 선형 회귀 수식

입력 변수가 $p$개일 때:

$$\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_px_p + b$$

단순 회귀와 본질은 같다. 직선 하나 대신 **고차원 평면(hyperplane)**을 데이터에 맞추는 것이다.

---

## 2. 행렬로 표현하기

변수가 많아질수록 수식이 길어진다. 이를 간결하게 표현하는 것이 행렬이다.

데이터 $n$개, 변수 $p$개가 있을 때:

$$\mathbf{X} = \begin{bmatrix} 1 & x_{11} & \cdots & x_{1p} \\ 1 & x_{21} & \cdots & x_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & \cdots & x_{np} \end{bmatrix}, \quad \mathbf{w} = \begin{bmatrix} b \\ w_1 \\ \vdots \\ w_p \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$$

- $\mathbf{X}$ : 입력 행렬 ($n \times (p+1)$), 첫 열은 절편을 위한 1
- $\mathbf{w}$ : 가중치 벡터
- $\mathbf{y}$ : 실제값 벡터

예측값을 한 줄로 표현하면:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$$

---

## 3. 행렬로 OLS 풀기

MSE를 행렬로 표현하고 미분하면 OLS 해(정규방정식, Normal Equation)가 나온다:

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

단 한 줄로 최적의 가중치를 구할 수 있다.

```python
import numpy as np

# 데이터: 면적(x1), 방 개수(x2) → 집값(y)
X_raw = np.array([
    [60, 2],
    [80, 3],
    [100, 3],
    [120, 4],
    [150, 4],
])
y = np.array([200, 280, 330, 400, 470])

# 절편을 위한 1 열 추가
X = np.hstack([np.ones((len(X_raw), 1)), X_raw])

# 정규방정식
w = np.linalg.inv(X.T @ X) @ X.T @ y

print(f"절편: {w[0]:.2f}")
print(f"면적 가중치: {w[1]:.2f}")
print(f"방 개수 가중치: {w[2]:.2f}")
```

---

## 4. 주의할 점 — 다중공선성

입력 변수들 사이에 강한 상관관계가 있으면 $(\mathbf{X}^T\mathbf{X})^{-1}$ 계산이 불안정해진다.
이를 **다중공선성(Multicollinearity)**이라 한다.

예를 들어 "면적"과 "평수"를 동시에 넣으면 사실상 같은 정보가 두 번 들어가는 것이다.

확인 방법: **VIF(분산팽창인수)** 계산

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X_raw, i) for i in range(X_raw.shape[1])]
print(vif)  # 10 이상이면 다중공선성 의심
```

---

## 정리

| 개념 | 설명 |
|---|---|
| 다중 선형 회귀 | 입력 변수가 여러 개인 선형 회귀 |
| 행렬 표현 | $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ |
| 정규방정식 | $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ |
| 다중공선성 | 변수 간 높은 상관관계, VIF로 확인 |

다음 글: **과적합과 정규화 — Ridge, Lasso, ElasticNet**
