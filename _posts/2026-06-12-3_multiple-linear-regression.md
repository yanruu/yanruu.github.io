---
title: "[회귀분석 #3] 다중 선형 회귀와 행렬 표현"
date: 2026-06-12 09:20:00 +0900
slug: multiple-linear-regression
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

변수가 하나였던 단순 선형 회귀를 여러 개로 확장하고, 행렬로 깔끔하게 표현하는 법과 변수를 잘못 선택했을 때 생기는 다중공선성 문제를 다룬다.

[지난 글](../ols-gradient-descent)에서는 공부 시간 하나로 시험 점수를 예측했다.
하지만 현실에서 집값을 예측한다고 생각해보자.
면적만 본다고 정확한 집값이 나올까? 방 개수, 연식, 위치도 당연히 영향을 준다.
이처럼 **입력 변수가 여러 개인 경우**가 **다중 선형 회귀(Multiple Linear Regression)**다.

---

## 1. 단순 → 다중으로 확장

단순 선형 회귀가 2D 평면에 직선을 긋는 것이었다면,
다중 선형 회귀는 **고차원 공간에 평면(hyperplane)을 맞추는 것**이다.

입력 변수가 $p$개일 때 수식은:

$$\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_px_p + b$$

예를 들어 집값 예측에 3가지 변수를 쓴다면:

$$\hat{\text{집값}} = w_1 \cdot \text{면적} + w_2 \cdot \text{방 개수} + w_3 \cdot \text{연식} + b$$

각 $w$는 **해당 변수가 집값에 미치는 영향력**이다.
- $w_1 = 2.8$ 이면 면적이 1m² 늘어날 때 집값이 2.8만원 오른다
- $w_3 = -1.5$ 이면 연식이 1년 늘어날 때 집값이 1.5만원 내려간다

변수가 늘어날수록 모델이 더 많은 정보를 반영해 예측이 정확해진다.

---

## 2. 왜 행렬로 표현하는가

변수가 10개, 100개가 되면 위 수식은 엄청나게 길어진다.
행렬을 쓰면 변수가 몇 개든 **단 한 줄**로 표현할 수 있다.

데이터 $n$개, 변수 $p$개일 때:

$$\mathbf{X} = \begin{bmatrix} 1 & x_{11} & \cdots & x_{1p} \\ 1 & x_{21} & \cdots & x_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & \cdots & x_{np} \end{bmatrix}, \quad \mathbf{w} = \begin{bmatrix} b \\ w_1 \\ \vdots \\ w_p \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$$

- $\mathbf{X}$ : 입력 행렬 — 첫 열은 절편 $b$를 위해 1로 채움
- $\mathbf{w}$ : 가중치 벡터 — 우리가 찾아야 할 값
- $\mathbf{y}$ : 실제값 벡터

예측값 전체를 한 번에 계산하면:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$$

---

## 3. 정규방정식 — 행렬로 OLS 풀기

[2장](../ols-gradient-descent)에서 단순 회귀의 OLS를 배웠다.
다중 회귀에서도 같은 원리로 MSE를 행렬 미분하면 **정규방정식(Normal Equation)**이 나온다:

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

변수가 아무리 많아도 이 수식 하나로 최적의 $\mathbf{w}$를 한 번에 구할 수 있다.

```python
import numpy as np

# 데이터: 면적(m²), 방 개수, 연식 → 집값(만원)
X_raw = np.array([
    [60,  2, 15],
    [80,  3, 10],
    [100, 3,  5],
    [120, 4,  8],
    [150, 4,  3],
    [90,  2, 20],
    [110, 3, 12],
    [75,  2,  7],
])
y = np.array([200, 280, 330, 400, 470, 240, 350, 230])

# 절편을 위한 1 열 추가
X = np.hstack([np.ones((len(X_raw), 1)), X_raw])

# 정규방정식으로 최적 가중치 계산
w = np.linalg.inv(X.T @ X) @ X.T @ y

print(f"절편     : {w[0]:.2f}")
print(f"면적 가중치 : {w[1]:.2f}")   # 면적 1m² 증가 → 집값 약 w[1]만원 상승
print(f"방 개수 가중치: {w[2]:.2f}")
print(f"연식 가중치 : {w[3]:.2f}")   # 음수 → 오래될수록 집값 하락
```

결과를 보면 연식 가중치가 음수로 나온다.
오래된 집일수록 집값이 낮다는 현실 상식과 일치하는 결과다.
이처럼 다중 회귀의 가중치는 **각 변수의 영향력 방향과 크기**를 동시에 알려준다.

---

## 4. 다중공선성 — 같은 정보를 두 번 넣으면?

### 직관적으로 이해하기

다중공선성(Multicollinearity)이란 **입력 변수들 사이에 강한 선형 관계가 있는 것**이다.

쉬운 비유로 생각해보자.
집값을 예측하는데 "면적(m²)"과 "평수"를 둘 다 변수로 넣었다고 하자.
1평 = 3.305m²이므로 평수는 면적을 3.305로 나눈 것과 완전히 같은 정보다.

모델 입장에서는 이렇게 된다:
> "면적이 집값에 영향을 주는 건지, 평수가 주는 건지 구분이 안 돼. 둘이 완전히 같은 말을 하고 있잖아."

결과적으로 $w_{\text{면적}}$과 $w_{\text{평수}}$에 이상한 값이 들어가기 시작한다.
어느 변수에 얼마나 비중을 줄지 모델이 결정하지 못하기 때문이다.

### 수치적으로 왜 문제인가

정규방정식 $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ 에서
두 변수가 완전히 동일하면 $\mathbf{X}^T\mathbf{X}$ 행렬의 역행렬이 존재하지 않는다.

역행렬이 없다는 것은 "정답이 무한히 많다"는 뜻이다.
$w_{\text{면적}} = 3, w_{\text{평수}} = 0$ 이나 $w_{\text{면적}} = 0, w_{\text{평수}} = 9.9$ 나
예측값이 똑같이 나오기 때문에 모델이 어느 값을 선택해야 할지 알 수 없다.

### 시각화로 확인

아래 그래프에서 면적과 평수의 관계(오른쪽)는 완벽한 직선이다.
반면 면적과 방 개수(왼쪽)는 어느 정도 상관은 있지만 완전히 같지는 않다.

![다중공선성 직관](/assets/images/collinearity_scatter.png)

상관계수 히트맵으로 전체 변수 간 관계를 한눈에 볼 수 있다.
면적과 평수의 상관계수가 1.00으로 완전한 다중공선성을 보인다.

![상관계수 히트맵](/assets/images/multicollinearity_heatmap.png)

### 평수를 추가해도 성능이 안 오르는 이유

아래 그래프를 보면 면적+방 개수+연식 조합과 거기에 평수까지 추가한 조합의 R²가 동일하다.
같은 정보를 추가했기 때문에 새로운 정보가 없어 성능 향상이 없는 것이다.

![R² 비교](/assets/images/r2_comparison.png)

### VIF로 다중공선성 확인

**VIF(Variance Inflation Factor, 분산팽창인수)**는 각 변수가 다른 변수들로 얼마나 설명되는지를 나타낸다.

- VIF = 1 : 다른 변수와 무관
- VIF 5~10 : 다중공선성 주의
- VIF > 10 : 다중공선성 심각, 해당 변수 제거 고려

```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.DataFrame(X_raw, columns=['면적', '방개수', '연식'])

# 평수 추가 (다중공선성 유발)
df['평수'] = df['면적'] / 3.305

vif_data = pd.DataFrame()
vif_data['변수'] = df.columns
vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
print(vif_data)
# 면적과 평수의 VIF가 매우 높게 나옴
```

### 해결 방법

- **변수 제거**: 면적과 평수 중 하나만 사용
- **도메인 지식 활용**: 실제로 의미가 다른 변수만 선택
- **Ridge 정규화**: 다음 글에서 자세히 다룬다

---

## 5. 어디에 쓰이는가

다중 선형 회귀는 **"여러 요인이 결과에 미치는 영향을 동시에 분석"** 할 때 가장 많이 쓰인다.

- **부동산**: 면적, 층수, 위치, 연식으로 집값 예측
- **마케팅**: 광고비, 노출 횟수, 클릭률로 매출 예측
- **의학/역학**: 나이, 체중, 흡연 여부로 혈압 예측
- **금융**: 여러 경제 지표로 주가 예측 (단, 시계열 특성 추가 고려 필요)

특히 **"어떤 변수가 결과에 얼마나 영향을 주는가"를 수치로 설명**해야 할 때,
딥러닝보다 다중 선형 회귀가 더 적합한 경우가 많다.

---

## 6. 직접 실험해보기

VS Code에서 `.ipynb`로 아래를 직접 실험해보자:

- 변수를 하나씩 추가할 때마다 R²가 어떻게 바뀌는지
- 면적과 평수를 동시에 넣으면 가중치가 어떻게 이상해지는지
- VIF를 계산해서 다중공선성이 있는 변수를 찾아보기

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(42)
n = 100
area   = np.random.uniform(50, 150, n)
pyeong = area / 3.305
rooms  = np.round(area / 35 + np.random.normal(0, 0.5, n)).clip(1, 5)
age    = np.random.uniform(1, 30, n)
price  = 2.8 * area + 10 * rooms - 1.5 * age + np.random.normal(0, 20, n)

# 변수 조합별 R² 비교
configs = {
    '면적만':              [area],
    '면적 + 방':           [area, rooms],
    '면적 + 방 + 연식':    [area, rooms, age],
    '면적 + 평수 + 방 + 연식 (다중공선성)': [area, pyeong, rooms, age],
}

for name, cols in configs.items():
    X = np.column_stack(cols)
    model = LinearRegression().fit(X, price)
    r2 = r2_score(price, model.predict(X))
    print(f"{name}: R² = {r2:.4f}")

# VIF 계산
df = pd.DataFrame({'면적': area, '평수': pyeong, '방개수': rooms, '연식': age})
for i, col in enumerate(df.columns):
    vif = variance_inflation_factor(df.values, i)
    print(f"VIF({col}) = {vif:.2f}")
```

---

## 정리

| 개념 | 설명 |
|---|---|
| 다중 선형 회귀 | 변수가 여러 개인 선형 회귀, 고차원 평면을 데이터에 맞춤 |
| 행렬 표현 | $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ — 변수가 많아도 한 줄로 표현 |
| 정규방정식 | $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ — 한 번에 최적해 계산 |
| 다중공선성 | 변수 간 강한 상관관계 → 역행렬 불안정, 가중치 신뢰 불가 |
| VIF | 다중공선성 수치 확인, 10 이상이면 해당 변수 제거 고려 |

다음 글: **[[회귀분석 #4] 과적합과 정규화 — Ridge, Lasso, ElasticNet](/ml/regularization/)**
