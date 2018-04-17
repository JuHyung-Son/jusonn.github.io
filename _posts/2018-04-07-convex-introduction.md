---
title: Convex function, Convex Optimization review
date: 2018-04-07
author: JuHyung Son
layout: post
tags:
   - 최적화
   - convex
categories:
   - MATH
---

# Convex Opt
최적화 분야에는 크게 Convex와 Non-Convex 최적화가 있습니다. 최적화하려는 대상에 따라 분야가 달라지기보다는 어떤 방법으로 최적화를 할 것이냐에 따라 분야가 나뉩니다. 왜냐면 Convex 최적화에서는 non-convex한 문제도 Convex한 형태로 변형하여 풀기 때문이죠. 하지만 Convex한 문제를 Non-convex한 문제로 풀 거 같지는 않네요. 왜냐면 일반적으로 Non-Convex한 문제는 최적화가 어렵고 거의 불가능합니다. 딥러닝에서도 global minimum 대신 적당히 좋은 local minimum을 사용하는 것 처럼요. 반면 Convex한 문제는 최적화가 비교적 훨씬 쉽습니다. 고등학생 때 이차함수를 미분하며 global minimum을 찾았던 거 처럼 말이죠. 이러한 것들은 대체적으로 쉬우니 Convex opt에서는 non-convex한 문제를 convex한 문제로 변환하는 것에 초점을 둡니다.

즉, 어떤 문제를 $P:min_{x \in D} f(x)$와 같은 형태로 바꾼 후 푸는 것입니다. P가 어떤 문제냐에 따라 필요한 알고리즘도 달라지고 Convex로 바꿀 수 있는 non-convex의 형태도 다양하기에 어떤 형태의 P 라는 문제 자체에 대한 이해가 있어야 합니다.

근데 대부분의 기계학습, 특히 딥러닝에서의 최적화 문제를 Convex로 바꿀 수 있는 지는 모르겠습니다. 오랫동안 연구된 기계학습 알고리즘은 이미 convex로 바꾸어 풀었거나 아니면 안되거나 일테고, 딥러닝의 loss는 convex로 바꾸지 못합니다. 그렇지만 Convex Optimization를 통해 최적화에 관한 어떤 인사이트를 얻을 수는 있는 거 같습니다.

그럼 먼저 Convex가 뭔지에 대해 먼저 알아야합니다. 한국말로 쉽게 말하면 아래로 볼록한 형태의 함수가 convex 함수입니다. 위로 볼록한 형태는 concave라고 하죠. Concave는 위아래로 뒤집으면 convex가 되기도 하고 실제 대부분의 문제는 어떤 비용을 최소화하는 것이기 때문에 convex가 더 대표적으로 쓰입니다.

### Convex set
>$C \subset \mathbb{R}^{n}$ is a convex set if $x,y \in C \rightarrow tx+(1-t)y \in C$ for all $0 \leq t \leq 1$

어떤 집합 C의 원소들이 위와 같은 연산을 했을 때 다시 집합 C에 속한다면 convex set입니다. 또 $tx+(1-t)y$와 같은 점을 점 x,y,의 convex combination 이라고 합니다.
<div align='center'> <img src="/image/convex/1.jpg"/> </div>
#### Ex
<li>
   All of $\mathbb{R}^n$ </br>
   두 실수의 연산이 실수에 포함된다는 건 너무 자명하죠.
</li>
<li>
   convex 집합들의 교집합. </br>
   $C_1 , ..., C_k 의 convex 집합이 있을 때, 이것들의 교집합인 $\bigcap C_i = \{ x: x \in C \}$도 convex 집합입니다.
</li>

### Convex function

Convex Optimization의 중심에 있는 convex function입니다.

>$f : \mathbb{R}^{n} \rightarrow \mathbb{R}$ is a **convex function** if $dom(f) \subset \mathbb{R}^{n}$ is convex, and $f(tx + (1-t)y) \leq t f(x) + (1-t)f(y)$ for all $0 \leq t \leq 1$ and all $x,y \in dom(f)$

Convex function은 정의역을 실수로 보내는 집합입니다. 그리고 정의역이 convex 집합이고 위의 조건이 만족한다면 f를 최적화 함수라고 합니다. 정의역이 convex인 조건은 치역이 존재하기 위함을 확인하는 것이라 그다지 신경쓰지 않아도 된다네요. 그리고 $f(tx + (1-t)y) \leq t f(x) + (1-t)f(y)$은 함수 f가 (x,y) 구간에서 f(x), f(y)를 이은 직선 보다 작다는 것이니 아래로 볼록하다는 것을 뜻합니다.
<div align='center'> <img src="/image/convex/2.jpg"/> </div>

#### First Order Condition for Convexity

위 그림과 같이 이차 함수가 아래로 볼록하다면 함수 위의 아무 점에서의 접선은 항상 함수보다 아래 있겠죠? 이건 일차원에서의 설명이고 n차원에서도 수학적으로 쉽게 표현할 수 있습니다. 함수 $f(x)$에서의 기울기는 $\triangledown_{x} f(x)$로 표기합니다. 그렇다면 $f(x)$에서의 접선은 $f(x)+ \triangledown f(x)^T (y-x)$로 나타내고 이 직선은 임의의 함수값 $f(y)$ 보다 항상 작으니 밑의 식으로 나타낼 수 있죠. $$f(y) \geq f(x) + \triangledown_x f(x)^T (y-x)$$
이것을 First Order Condition for Convexity 라고 합니다.

#### Second Order Condition for Convexity

고등학교에서 함수의 두번 미분한 값이 0보다 크면 기울기가 증가하는 것이라고 배웠습니다. 이것을 이용해 minima를 찾았던 게 기억나네요. 이건 n차원에서도 같습니다. $f$를 $x \in D(f)$에서 두번 미분한 Hessian $\triangledown_x^2 f(x) \succ 0$ 이라면 $f$는 convex 입니다. $\succ 0$ 는 행렬이 positive semidefinite 하다는 걸 말합니다.

#### Jensen's Inequality

통계학에서 자주 보이는 Jensen's Inequality 는 쉽게 말해 convex 함수의 두 점을 잇는 직선은 두 점 사이에서 convex 함수 위에 있다는 정리입니다. convex 함수의 기본 정의에서 시작해서 $$f( \mathbb{E} [x]) \leq \mathbb{E} [f(x)]$$ 임을 증명하고 이 부등식이 Jensen's Inequality입니다.

#### Sublevel sets

$\alpha$-Sublevel set은 $f(x) \leq \alpha$ 를 만족하는 모든 $x$의 집합입니다. 함수에서 어떤 구간을 나타내겠죠. 정의는 다음과 같습니다. $$f: \mathbb{R}^n \rightarrow R ~ \alpha \in \mathbb{R}$$ $$\{x \in D(f) : f(x) \leq \alpha \}$$
convex 함수의 구간을 자른 것이니 convex임이 당연하게 보입니다. 그러니 증명은 패스합니다.

### Ex

우리가 익히 알고 있는 이런 함수들이 convex 입니다.
- Exponential
- $-log(x)$
- Affine 함수
- Quadratic 함수
- Norm
위 함수들은 이미 많이 봐왔고 그래프를 그려보면 딱봐도 Convex입니다. 이것 말고 다른 것들이 나오기 시작하면 좀 어려워집니다.

##Convex Optimization Problem

자 convex opt의 필수 개념을 알았습니다. Convex set 과 convex function 말이죠. 이제 이것을 이용해 무슨 문제를 푼다는 건지 알아야겠죠. Convex opt problem은 최적화 문제를 푸는 방법 중의 하나입니다. Convex한 성질을 이용하기 때문에 convex opt 죠. 일반적인 최적화 문제는 이렇게 생겼습니다.
$$minimize ~ f(x)$$
$$subject ~ to ~ x \in C $$
여기서 함수 f는 convex function이고 집합 C는 convex set 이라 convex optimization이 되는겁니다. 그리고 보통 조건 부분은 위에 식보다는 아래처럼 자세히 쓰는 것이 일반적입니다.
$$subject ~ to ~ g_{i} (x) \leq 0 ,~ i=1,...,m$$
$$h_{i}(x)=0, ~ i = 1,...,p$$
여기서의 g는 convex function이고 h는 affine function이죠.
그런데 각 g는 왜 0 이하이고 h는 0과 같을까요? 위에서 $\alpha - sublevel ~ set$을 봤었죠. 그걸 기억해보면 $g_{i}$의 0-sublevel set은 convex set이죠. 그리고 convex set들의 교집합은 convex set입니다. 반면 어떤 집합 $g_{i} \geq 0$ 이라면 그건 더이상 convex set이 아니게 됩니다. 따라서 그것으로 찾은 답(minimum)이 global minimum이라는 것을 보증하지 못하게 됩니다. 다음으로, 등호는 오직 affine function에서만 성립하기에 h는 affine function입니다.
이런 문제에서 찾은 $f(x)$ 값을 **optimal value** 라고 하고 $p^*$로 표기합니다. 식으로는 다음처럼 표현하죠.
$$p^* = min\{  f(x) : g_{i}(x) \leq 0 , i = 1, … , m, ~ h_{i} (x) = 0 , i=1,…, p \}$$
그리고 이런 $p^*$로 가는 $x$를 **optimal point**라고 하고
$$x^*$$ 로 표기합니다.

## Global Optimal

Convex 를 공부하는 건, 이 Convex의 성질이 $p^*$를 찾는데 아주 중요하기 때문입니다. 어떤 문제가 Convex 라면 그것의 모든 loacal optimal은 사실 global optimal입니다. 이 부분에 대한 증명은
<a href=“http://web.stanford.edu/class/cs224n/readings/cs229-cvxopt.pdf”>stanford 강의노트</a> 9페이지에 잘 나와있습니다.
증명이 따로 필요없다면 Global optimal과 local optimal의 부터 정의를 봅시다.

> **Def**
> A point x is **locally optimal** if it is feasible and if there exists some $R > 0$ such that all feasible points $z$ with $||x-z||_{2} \leq R$ satisfy $f(x) \leq f(z)$

쉽게 말해 x에서 R 만큼의 모든 점 z에서 x의 함수값이 z의 함수값보다 작으면, 즉 아래로 볼록하면 그 $x$는 locally optimal 입니다.
그리고 **Global optimal**은 x에서 R만큼의 z가 아닌 feasible한 모든 z에 대해서
$$f(x) \leq f(z)$$ 가 성립하여야 합니다.
Global optimal은 local optimal의 정의로부터 나오죠.

참조
Stanford - Convex Optimization review
