---
title: "[회귀분석 #2] OLS와 경사하강법으로 최적의 직선 찾기"
date: 2026-06-12 09:10:00 +0900
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

> 선형 회귀에서 최적의 $w$, $b$를 찾는 두 가지 방법 — 수식으로 단번에 구하는 OLS, 반복적으로 찾아가는 경사하강법을 비교한다.

[지난 글](../linear-regression-intro)에서 선형 회귀의 목표는 **MSE를 최소화하는 $w$와 $b$를 찾는 것**이라고 했다.
그런데 어떻게 찾을까? 방법이 두 가지다.

- **OLS**: 수학적으로 수식 한 번에 직접 구한다
- **경사하강법**: 조금씩 조금씩 반복하며 찾아간다

두 방법 모두 같은 목표를 향하지만, 접근 방식이 완전히 다르다.
어떤 상황에 어떤 방법을 써야 하는지까지 이 글에서 정리한다.

---

## 1. OLS (최소제곱법)

### 직관적으로 이해하기

OLS는 **Ordinary Least Squares**, 우리말로 최소제곱법이다.
"Least Squares"라는 이름처럼 **잔차(오차)의 제곱 합을 최소화**하는 직선을 찾는다.

잔차(Residual)란 실제값과 예측값의 차이다.

$$\text{잔차}_i = y_i - \hat{y}_i$$

아래 그래프에서 점선이 바로 잔차다.
OLS는 이 점선들의 길이를 제곱해서 더한 값(MSE)이 가장 작아지는 직선을 수학적으로 찾아낸다.

![OLS 잔차 시각화](/assets/images/ols_residual.png)

각 점에서 회귀선까지의 거리(점선)가 최소가 되도록 직선이 결정된다.
점들이 회귀선 위아래로 고르게 분포하고, 어느 한쪽으로 치우치지 않는 것이 잘 맞은 직선의 특징이다.

### 수식 유도

MSE를 $w$와 $b$에 대해 편미분하고 0으로 놓으면 닫힌 형태(closed-form)의 해가 나온다.

$$w = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

$$b = \bar{y} - w\bar{x}$$

- $\bar{x}$ : $x$의 평균, $\bar{y}$ : $y$의 평균
- 분자: $x$와 $y$가 함께 변하는 정도 (공분산과 유사)
- 분모: $x$가 퍼진 정도 (분산과 유사)

즉 **$w$는 x의 변화량 대비 y의 변화량 비율**이다. 직관적으로도 맞는 수식이다.

### 코드로 확인

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([2.1, 3.8, 4.5, 5.9, 6.2, 7.8, 8.5, 9.1], dtype=float)

x_mean, y_mean = x.mean(), y.mean()
w = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b = y_mean - w * x_mean

print(f"w = {w:.4f}")  # 0.9845
print(f"b = {b:.4f}")  # 1.5571
```

단 몇 줄로 최적의 $w$와 $b$를 바로 구할 수 있다.
수식이 닫힌 형태이기 때문에 반복 없이 **1회 계산으로 정확한 정답**이 나온다.

### OLS는 어디에 쓰이는가

- **소규모 데이터 분석**: 행 수가 수만 건 이하이고 변수가 많지 않을 때
- **통계적 해석이 필요한 경우**: 회귀계수의 신뢰구간, p-value 계산이 필요할 때
- **경제학, 사회과학 연구**: "변수 A가 변수 B에 미치는 영향"을 정량화할 때
- **베이스라인 모델**: 복잡한 모델 쓰기 전에 먼저 OLS로 기준선을 잡을 때

---

## 2. 경사하강법 (Gradient Descent)

### 직관적으로 이해하기

경사하강법은 **산에서 가장 가파른 내리막 방향으로 한 걸음씩 내려가는 것**과 같다.

지금 서 있는 위치(현재 $w$, $b$)에서 MSE가 줄어드는 방향을 계산하고, 그 방향으로 조금 이동한다.
이걸 수백~수천 번 반복하다 보면 결국 MSE가 최소인 지점에 도달한다.

### 업데이트 규칙

$$w \leftarrow w - \alpha \cdot \frac{\partial MSE}{\partial w}$$

$$b \leftarrow b - \alpha \cdot \frac{\partial MSE}{\partial b}$$

- $\alpha$ : **학습률(learning rate)** — 한 번에 얼마나 이동할지 결정하는 핵심 하이퍼파라미터
  - 너무 크면: 최솟값을 지나쳐 오히려 발산할 수 있음
  - 너무 작으면: 수렴 속도가 느려 학습 시간이 길어짐
- 편미분이 양수면 현재 $w$가 너무 크다는 뜻 → $w$를 줄이는 방향으로 이동

편미분을 전개하면:

$$\frac{\partial MSE}{\partial w} = -\frac{2}{n}\sum_{i=1}^{n}x_i(y_i - \hat{y}_i), \quad \frac{\partial MSE}{\partial b} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)$$

### Loss 감소 확인

아래 그래프는 경사하강법이 진행되면서 MSE가 어떻게 변하는지 보여준다.

![경사하강법 Loss 감소 곡선](/assets/images/gradient_descent_loss.png)

처음에는 MSE가 매우 높지만, 반복할수록 빠르게 감소하다가 일정 수준에서 수렴한다.
초반 20~30 epoch에서 대부분의 학습이 이루어지고, 이후에는 미세 조정이 계속된다.

### w, b 수렴 과정

$w$와 $b$ 값이 반복을 통해 OLS 정답에 수렴하는 과정을 직접 확인할 수 있다.

![w, b 수렴 과정](/assets/images/gradient_descent_convergence.png)

충분히 반복하면 경사하강법의 결과가 OLS의 정확한 해에 수렴하는 것을 볼 수 있다.
이는 두 방법이 **같은 문제(MSE 최소화)를 다른 방식으로 푸는 것**임을 보여준다.

### 코드로 확인

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([2.1, 3.8, 4.5, 5.9, 6.2, 7.8, 8.5, 9.1], dtype=float)

w, b = 0.0, 0.0
lr = 0.01
n = len(x)

for epoch in range(1000):
    y_pred = w * x + b
    dw = -2/n * np.sum(x * (y - y_pred))
    db = -2/n * np.sum(y - y_pred)
    w -= lr * dw
    b -= lr * db

print(f"w = {w:.4f}")  # ≈ 0.9845
print(f"b = {b:.4f}")  # ≈ 1.5571
```

1000번 반복 후 OLS와 거의 동일한 결과가 나온다.

### 경사하강법은 어디에 쓰이는가

- **대용량 데이터**: 수백만 행이 넘는 데이터에서 OLS의 행렬 연산은 메모리 부담이 큼
- **딥러닝**: LSTM, Transformer 같은 복잡한 모델은 OLS로 풀 수 없음. 경사하강법만 가능
- **온라인 학습**: 데이터가 실시간으로 들어올 때 조금씩 업데이트하며 학습
- **비선형 모델**: 닫힌 형태의 해가 없는 복잡한 모델에서 유일한 최적화 방법

---

## 3. OLS vs 경사하강법 — 언제 뭘 쓸까

| | OLS | 경사하강법 |
|---|---|---|
| 방식 | 수식 1회 계산 | 수백~수천 번 반복 |
| 속도 | 빠름 | 상대적으로 느림 |
| 데이터 크기 | 소규모에 유리 | 대규모에 유리 |
| 정확도 | 정확한 해 | 근사해 (수렴에 따라 다름) |
| 적용 범위 | 선형 모델만 | 비선형, 딥러닝까지 |
| 하이퍼파라미터 | 없음 | 학습률($\alpha$) 필요 |
| 주요 사용처 | 통계 분석, 소규모 회귀 | 딥러닝, 대규모 데이터 |

**결론:**
- 데이터가 작고 선형 모델이면 → **OLS**
- 데이터가 크거나 딥러닝으로 확장할 계획이면 → **경사하강법**
- LSTM, TFT 같은 모델은 경사하강법의 직접적인 확장선에 있다

---

## 직접 실험해보기

VS Code에서 `.ipynb` 파일로 실행하면서 아래를 직접 확인해보자:

- 학습률(`lr`)을 0.1, 0.001로 바꾸면 수렴 속도가 어떻게 달라지는지
- epoch를 100, 500, 2000으로 바꾸면 결과가 어떻게 달라지는지
- OLS와 경사하강법의 최종 $w$, $b$ 값이 얼마나 차이 나는지

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([2.1, 3.8, 4.5, 5.9, 6.2, 7.8, 8.5, 9.1], dtype=float)

# OLS
x_mean, y_mean = x.mean(), y.mean()
w_ols = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b_ols = y_mean - w_ols * x_mean

# 경사하강법
w_gd, b_gd = 0.0, 0.0
lr = 0.01  # ← 이 값을 바꿔보자
losses = []

for epoch in range(500):  # ← 이 값도 바꿔보자
    y_pred = w_gd * x + b_gd
    loss = np.mean((y - y_pred) ** 2)
    losses.append(loss)
    dw = -2/len(x) * np.sum(x * (y - y_pred))
    db = -2/len(x) * np.sum(y - y_pred)
    w_gd -= lr * dw
    b_gd -= lr * db

print(f"OLS        → w={w_ols:.4f}, b={b_ols:.4f}")
print(f"경사하강법 → w={w_gd:.4f}, b={b_gd:.4f}")

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Loss 감소 곡선')
plt.show()
```

---

## 정리

| 개념 | 설명 |
|------|------|
| OLS | MSE를 미분해 단번에 최적 $w$, $b$를 구하는 방법 |
| 잔차 | 실제값과 예측값의 차이, OLS는 이 제곱합을 최소화 |
| 경사하강법 | 기울기 방향으로 조금씩 이동하며 최솟값을 찾는 방법 |
| 학습률 | 한 번에 이동하는 크기, 너무 크거나 작으면 문제 발생 |

다음 글: **[[회귀분석 #3] 다중 선형 회귀와 행렬 표현](/ml/multiple-linear-regression/)**
