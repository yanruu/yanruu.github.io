---
title: "[회귀분석 #4] 과적합과 정규화 — Ridge, Lasso, ElasticNet"
date: 2026-06-12 09:40:00 +0900
slug: regularization
categories:
  - ML
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

## 잠깐, "정규화"라는 말이 헷갈릴 수 있다

여기서 말하는 정규화는 선형대수에서 배우는 정규화(normalization)와는 다른 개념이다. 영어로는 원래 다른 단어(regularization vs normalization)인데, 한국어로 옮기면서 같은 표현을 쓰게 됐을 뿐이다.

선형대수의 정규화는 벡터 $v$를 그 벡터의 크기(norm)로 나눠서 길이가 1인 단위벡터로 만드는 것이다. 예를 들어 $v=(3,4)$의 L2 norm은 $\sqrt{3^2+4^2}=5$이고, 이 벡터를 정규화하면 $(3/5,\ 4/5)$가 된다 — 크기를 1로 만들고 방향만 남기는 과정이다.

이번 글의 정규화(regularization)는 가중치 벡터를 단위벡터로 만드는 게 아니다. 가중치 벡터의 크기(norm)를 계산해서, 그 값이 크면 손실함수에 벌점으로 더하는 것뿐이다. 다만 "크기를 하나의 숫자로 계산한다"는 도구 자체는 선형대수의 norm과 완전히 같다.

$$\text{L2 norm: } \|w\| = \sqrt{\sum_j w_j^2} \qquad \text{L1 norm: } \|w\|_1 = \sum_j |w_j|$$

뒤에 나올 Ridge의 페널티 $\sum w_j^2$는 L2 norm의 제곱이고, Lasso의 페널티 $\sum |w_j|$는 L1 norm 그 자체다. 즉 정규화(regularization)는 "가중치 벡터의 norm을 구해서 그 값을 줄이도록 손실함수에 압박을 가하는 것"이고, normalization은 "그 norm으로 나눠서 방향만 남기는 것"이다 — 같은 norm이라는 재료로 서로 다른 목적을 수행하는 셈이다.

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

### 왜 Lasso만 정확히 0이 될까

변수 하나만 있고 표준화된 경우로 단순화해서 보면, OLS 해를 $w_{ols}$라 할 때 Ridge와 Lasso의 해는 다음과 같은 닫힌 형태(closed form)를 가진다.

$$w_{ridge} = \frac{w_{ols}}{1+\lambda} \qquad\qquad w_{lasso} = \text{sign}(w_{ols}) \cdot \max(|w_{ols}|-\lambda,\ 0)$$

Ridge는 그냥 $w_{ols}$를 $(1+\lambda)$로 나누는 것뿐이다. $\lambda$가 아무리 커져도 분모가 커질 뿐이라, $w_{ols} \ne 0$이면 결과도 절대 정확히 0이 되지 않는다.

Lasso는 "$\lambda$만큼 깎고, 깎아서 음수가 되면 0에서 멈춘다"는 소프트 스레숄딩(soft-thresholding)이다. $\lambda$가 $|w_{ols}|$보다 커지는 순간 그 변수의 계수는 정확히 0이 된다.

이 차이는 두 페널티가 **작은 가중치를 얼마나 신경 쓰는지**에서 나온다. $w$가 점점 작아질 때 두 페널티 값이 어떻게 줄어드는지 비교해보면:

| $w$ | L2 벌점 ($w^2$) | L1 벌점 ($\lvert w \rvert$) |
|---|---|---|
| 3 | 9 | 3 |
| 1 | 1 | 1 |
| 0.1 | 0.01 | 0.1 |
| 0.01 | 0.0001 | 0.01 |

L2(Ridge)는 $w$가 작아질수록 벌점이 제곱으로 훨씬 빠르게 사라진다 — $w=0.01$이면 벌점은 0.0001로 거의 무시할 수준이라, 굳이 0까지 밀어붙일 동기가 없다. 반면 L1(Lasso)은 $w$가 얼마나 작든 벌점이 그 값에 그대로 정비례한다 — $w=0.01$이어도 벌점 0.01은 여전히 의미가 있어서, 이 변수가 데이터를 맞추는 데 별 도움이 안 된다면 끝까지 깎아 정확히 0으로 만드는 쪽이 전체 손실을 더 줄이는 길이 된다.

기하학적으로 보면 더 분명해진다. Ridge의 제약조건 $\sum w_j^2 \le t$는 원(또는 구)이라 모서리가 없다. Lasso의 제약조건 $\sum |w_j| \le t$는 마름모(다이아몬드)라 좌표축 위에 모서리가 있다. 손실함수의 등고선(타원)이 이 도형에 처음 닿는 점이 정규화된 해가 되는데, 마름모는 축 위의 모서리에 닿을 확률이 높고 그 모서리에 닿으면 다른 변수의 계수가 정확히 0이 된다. 원에는 모서리가 없어 축 위에 정확히 닿을 일이 거의 없으므로 모든 계수가 작게나마 살아남는다.

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
