# Convex Opt
최적화 분야에는 크게 Convex와 Non-Convex 최적화가 있습니다. 최적화하려는 대상에 따라 분야가 달라지기보다는 어떤 방법으로 최적화를 할 것이냐에 따라 분야가 나뉩니다. 왜냐면 Convex 최적화에서는 non-convex한 문제도 Convex한 형태로 변형하여 풀기 때문이죠. 하지만 Convex한 문제를 Non-convex한 문제로 풀 거 같지는 않네요. 왜냐면 일반적으로 Non-Convex한 문제는 최적화가 어렵고 거의 불가능합니다. 딥러닝에서도 global minimum 대신 적당히 좋은 local minimum을 사용하는 것 처럼요. 반면 Convex한 문제는 최적화가 비교적 훨씬 쉽습니다. 고등학생 때 이차함수를 미분하며 global minimum을 찾았던 거 처럼 말이죠. 이러한 것들은 대체적으로 쉬우니 Convex opt에서는 non-convex한 문제를 convex한 문제로 변환하는 것에 초점을 둡니다.

즉, 어떤 문제를 $P:min_{x \in D} f(x)$와 같은 형태로 바꾼 후 푸는 것입니다. P가 어떤 문제냐에 따라 필요한 알고리즘도 달라지고 Convex로 바꿀 수 있는 non-convex의 형태도 다양하기에 어떤 형태의 P 라는 문제 자체에 대한 이해가 깊어야합니다.

그럼 먼저 Convex가 뭔지에 대해 먼저 알아야합니다. 한국말로 쉽게 말하면 아래로 볼록한 형태의 함수가 convex 함수입니다. 위로 볼록한 형태는 concave라고 하죠. Concave는 위아래로 뒤집으면 convex가 되기도 하고 실제 대부분의 문제는 어떤 비용을 최소화하는 것이기 때문에 convex가 더 대표적으로 쓰입니다.
이제 convex 집합과 함수의 정의를 보면 이렇습니다.

### Convex set
>$C \subset \mathbb{R}^{n}$ is a convex set if $x,y \in C \rightarrow tx+(1-t)y \in C$ for all $0 \leq t \leq 1$

어떤 집합 C의 원소들이 위와 같은 연산을 했을 때 다시 집합 C에 속한다면 convex set입니다.
### Convex function

>$f : \mathbb{R}^{n} \rightarrow \mathbb{R}$ is a **convex function** if $dom(f) \subset \mathbb{R}^{n}$ is convex, and $f(tx + (1-t)y) \leq t f(x) + (1-t)f(y)$ for all $0 \leq t \leq 1$ and all $x,y \in dom(f)$

Convex function은 정의역을 실수로 보내는 집합입니다. 그리고 정의역이 convex 집합이고  위의 조건이 만족한다면 f를 최적화 함수라고 합니다. 그리고 $f(tx + (1-t)y) \leq t f(x) + (1-t)f(y)$은 함수 f가 (x,y) 구간에서 f(x), f(y)를 이은 직선 보다 작다는 것이니 아래로 볼록하다는 것을 뜻합니다.

### Convex Optimization Problem

최적화 문제는 다음처럼 표기합니다.
$$min_{x \in D} ~~ f(x)$$
$$subject to ~~ g_{i}(x) \leq 0 , i=1,..,m$$
$$~~ ~~ h_{i}(x) = 0 , j=1,...,r$$
1. 여기서 $f, g$는 convex 함수여야 하고
2. $h$는 affine해야 합니다. Affine은 $ax+b$와 같은 일차함수를 말합니다.

### Local, Global Minima

그리고 이제 최적화에서 가장 중요한 minima를 봅니다. Minima에는 local minima와 global minima가 있습니다. 직관적으로 minima란 아래로 볼록한 부분을 말하고 local은 그런 부분 하나하나를 모두 뜻하며 global은 그 중에서 가장 값이 작은 부분은 뜻합니다. 이것을 수학적으로 정의하면,
어떤 x가 feasible할 때 함수 f를 local neighborhood에서 최소화한다면, $$f(x) \leq f(y) ~ for ~ all ~ feasible~y, ||x-y||_{2} \leq \epsilon$$
즉 x의 함수값이 x를 기준으로 엡실론 만큼의 주변 값의 함수값보다 작다면 f(x)는 local minima 입니다.
위의 정의에서 엡실론 대신 도메인 전체가 된다면 f(x)는 함수값들 중 가장 작은 값이 되고 즉, global minima 입니다.
