---
title: "[회귀분석 #4] 과적합과 정규화 — Ridge, Lasso, ElasticNet"
categories:
  - ML
tags:
  - 회귀분석
  - 정규화
  - Ridge
  - Lasso
  - ElasticNet
  - 과적합
toc: true
toc_sticky: true
---

## 들어가며

모델이 훈련 데이터에는 완벽하게 맞지만 새로운 데이터에는 형편없는 경우가 있다.
이것이 **과적합(Overfitting)**이다.
정규화(Regularization)는 모델이 너무 복잡해지지 않도록 제약을 거는 기법이다.

---

## 1. 과적합이란

아래 두 경우를 생각해보자:

- **과소적합(Underfitting)**: 모델이 너무 단순해서 데이터의 패턴을 못 잡음
- **과적합(Overfitting)**: 모델이 훈련 데이터의 노이즈까지 외워버림

회귀에서 과적합은 주로 변수가 많거나 가중치 $w$가 지나치게 커질 때 발생한다.

---

## 2. 정규화의 원리

기본 MSE에 **페널티 항**을 추가해서 가중치가 커지는 것을 억제한다.

$$\text{Loss} = MSE + \lambda \cdot \text{penalty}$$

- $\lambda$ : 정규화 강도 (클수록 가중치를 더 강하게 억제)

---

## 3. Ridge (L2 정규화)

페널티로 가중치의 **제곱합**을 사용한다:

$$\text{Loss}_{Ridge} = MSE + \lambda \sum_{j=1}^{p} w_j^2$$

- 가중치를 0에 가깝게 만들지만 **완전히 0으로 만들지는 않는다**
- 모든 변수를 조금씩 살려두는 방식
- 다중공선성이 있을 때 효과적

```python
from sklearn.linear_model import Ridge
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = Ridge(alpha=1.0)  # alpha = λ
model.fit(X, y)
print(f"w = {model.coef_[0]:.4f}, b = {model.intercept_:.4f}")
```

---

## 4. Lasso (L1 정규화)

페널티로 가중치의 **절댓값 합**을 사용한다:

$$\text{Loss}_{Lasso} = MSE + \lambda \sum_{j=1}^{p} |w_j|$$

- 중요하지 않은 변수의 가중치를 **완전히 0으로 만든다**
- 자동으로 변수 선택(Feature Selection) 효과
- 변수가 많고 일부만 중요할 때 유용

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X, y)
print(f"w = {model.coef_[0]:.4f}, b = {model.intercept_:.4f}")
```

---

## 5. ElasticNet (L1 + L2)

Ridge와 Lasso를 결합한 방식:

$$\text{Loss}_{EN} = MSE + \lambda_1 \sum|w_j| + \lambda_2 \sum w_j^2$$

- Ridge의 안정성 + Lasso의 변수 선택 효과
- 변수가 많고 서로 상관관계도 있을 때 사용

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio: L1 비중
model.fit(X, y)
print(f"w = {model.coef_[0]:.4f}, b = {model.intercept_:.4f}")
```

---

## 6. λ는 어떻게 정하나

$\lambda$가 너무 크면 과소적합, 너무 작으면 과적합이 된다.
**교차검증(Cross Validation)**으로 최적의 $\lambda$를 찾는다.

```python
from sklearn.linear_model import RidgeCV

model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
model.fit(X, y)
print(f"최적 lambda: {model.alpha_}")
```

---

## 정리

| | Ridge | Lasso | ElasticNet |
|---|---|---|---|
| 페널티 | $\sum w_j^2$ | $\sum \|w_j\|$ | L1 + L2 |
| 가중치 | 0에 가깝게 | 완전히 0 가능 | 둘의 중간 |
| 변수 선택 | X | O | O |
| 언제 쓰나 | 다중공선성 | 변수가 많을 때 | 둘 다 해당할 때 |

다음 글: **모델 성능 평가 — R², RMSE, MAE, MAPE**
