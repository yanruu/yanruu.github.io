---
title: "[회귀분석 #4] 과적합과 정규화 — Ridge, Lasso, ElasticNet"
date: 2026-06-12 09:40:00 +0900
slug: regularization
categories:
  - 머신러닝
tags:
  - 회귀분석
  - 정규화
  - Ridge
  - Lasso
  - ElasticNet
toc: true
toc_sticky: true
---

지난 글에서 다중회귀에서 변수가 늘어날수록 다중공선성과 과적합 위험이 커진다는 것을 확인했다. 이번 글에서는 이 문제를 직접 해결하는 방법, **정규화(Regularization)**를 다룬다. Ridge, Lasso, ElasticNet이 각각 어떤 수학적 장치로 과적합을 막는지, 그리고 왜 셋의 결과가 다르게 나오는지를 코드와 그래프로 확인한다.

## 과적합이란

모델이 학습 데이터의 패턴뿐 아니라 노이즈까지 외워버린 상태를 과적합(overfitting)이라 한다. 학습 데이터에 대한 오차는 계속 줄어들지만, 처음 보는 테스트 데이터에 대한 오차는 어느 시점부터 오히려 커진다.

다항회귀로 직접 확인해보자.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

np.random.seed(42)
x = np.sort(np.random.uniform(-3, 3, 40))
y_true = 0.5 * x**3 - x**2 + 2
y = y_true + np.random.normal(0, 3, len(x))

X_train, X_test, y_train, y_test = train_test_split(
    x.reshape(-1, 1), y, test_size=0.3, random_state=0
)

for d in [1, 3, 9]:
    model = make_pipeline(PolynomialFeatures(d), Ridge(alpha=1e-6))
    model.fit(X_train, y_train)
    train_rmse = np.sqrt(np.mean((model.predict(X_train) - y_train) ** 2))
    test_rmse = np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2))
    print(f"degree={d}: train RMSE={train_rmse:.2f}, test RMSE={test_rmse:.2f}")
```

차수를 1, 3, 9로 늘려가며 학습/테스트 오차를 비교하면, 차수가 커질수록 학습 오차는 계속 줄어들지만 테스트 오차는 9차에서 오히려 커진다 — 이것이 과적합이다.

![정규화 없이 모델 복잡도를 높였을 때와 정규화 강도에 따른 테스트 오차](/assets/images/regularization_train_test_error.png)
*왼쪽: 다항식 차수가 커질수록 학습 오차는 줄지만 테스트 오차는 다시 증가한다. 오른쪽: 같은 9차 모델에서도 λ(정규화 강도)를 조절하면 테스트 오차가 U자 곡선을 그린다.*

오른쪽 그래프가 정규화의 핵심을 보여준다. λ가 너무 작으면 정규화 효과가 없어 과적합이고, λ가 너무 크면 모델이 너무 단순해져 과소적합(underfitting)이다. 적절한 λ를 찾는 것이 정규화 튜닝의 본질이다.

## Ridge 회귀 (L2 정규화)

Ridge는 손실 함수에 가중치의 제곱합을 페널티로 추가한다.

$$L_{ridge} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} w_j^2$$

$\lambda$가 커지면 가중치들이 0에 가까워지지만, **정확히 0이 되지는 않는다.** 모든 변수를 조금씩 남겨두면서 크기를 줄이는 방식이다.

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print("Ridge coefficients:", ridge.coef_)
```

## Lasso 회귀 (L1 정규화)

Lasso는 절댓값 합을 페널티로 쓴다.

$$L_{lasso} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |w_j|$$

L1 페널티의 기하학적 특성 때문에 $\lambda$가 커지면 일부 가중치가 **정확히 0이 된다.** 즉 Lasso는 변수 선택(feature selection) 효과를 가진다 — 중요하지 않은 변수를 모델에서 완전히 제거한다.

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print("Lasso coefficients:", lasso.coef_)
print("0이 된 변수 개수:", np.sum(lasso.coef_ == 0))
```

## Ridge vs Lasso — 계수가 줄어드는 양상이 다르다

8개 변수(그중 절반은 실제로는 영향이 없는 변수)로 합성 데이터를 만들고, $\lambda$를 바꿔가며 각 변수의 계수가 어떻게 변하는지 추적했다.

![Ridge와 Lasso의 계수 경로 비교](/assets/images/ridge_lasso_coefficient_path.png)
*왼쪽 Ridge는 λ가 커져도 모든 계수가 살아있는 채로 천천히 줄어든다. 오른쪽 Lasso는 λ가 커지면 일부 계수가 정확히 0에서 멈춘다 — 변수가 모델에서 빠지는 것과 같다.*

이 그림이 Ridge와 Lasso의 본질적 차이를 가장 직관적으로 보여준다. 변수가 많고 그중 일부만 의미가 있다고 의심된다면 Lasso, 변수들이 서로 연관되어 있고 다 조금씩은 의미가 있다고 보면 Ridge가 유리하다.

## ElasticNet — 둘을 섞기

ElasticNet은 L1과 L2 페널티를 모두 사용한다.

$$L_{elastic} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \left( \alpha \sum_{j} |w_j| + (1-\alpha) \sum_{j} w_j^2 \right)$$

$\alpha$가 1이면 Lasso, 0이면 Ridge와 같아진다. 변수 간 상관관계가 강해서 Lasso가 변수를 불안정하게 골라낼 때, ElasticNet이 더 안정적인 결과를 주는 경우가 많다.

```python
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
print("ElasticNet coefficients:", elastic.coef_)
```

## λ를 어떻게 정할까 — RidgeCV / LassoCV

$\lambda$는 직접 하나씩 시도해서 정할 필요 없이 교차검증으로 자동 탐색할 수 있다.

```python
from sklearn.linear_model import RidgeCV, LassoCV

ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
ridge_cv.fit(X_train, y_train)
print("최적 alpha (Ridge):", ridge_cv.alpha_)

lasso_cv = LassoCV(alphas=np.logspace(-3, 3, 50), cv=5)
lasso_cv.fit(X_train, y_train)
print("최적 alpha (Lasso):", lasso_cv.alpha_)
```

## 정리

| 방법 | 페널티 | 계수가 0이 되는가 | 적합한 상황 |
|------|--------|-------------------|-------------|
| Ridge | L2 (제곱합) | 안 됨 | 변수들이 다 조금씩 중요, 다중공선성 완화 |
| Lasso | L1 (절댓값합) | 됨 | 변수 선택이 필요, 일부만 중요할 때 |
| ElasticNet | L1 + L2 | 됨 | 변수 간 상관관계가 강할 때 안정적 |

정규화는 "모델을 일부러 덜 정확하게 학습시켜서 더 일반화되게 만드는" 트레이드오프다. 다음 글에서는 이 트레이드오프를 어떻게 숫자로 측정하는지 — RMSE, MAE, MAPE 같은 평가 지표를 다룬다.

다음 글: **[모델 성능 평가 — R², RMSE, MAE, MAPE](/ml/evaluation-metrics/)**
