---
id: 849
title: Cross entropy, KL divergence
date: 2018-03-24T16:48:05+00:00
author: JuHyung Son
layout: post
permalink: /cross-entropy-kl-divergence/
image: /wp-content/uploads/2017/11/cropped-IMG_5429-1-250x250.jpg
categories:
  - Deep Learning
---
<h1>Information Theory</h1>
정보이론이라고 하는 Information theory는 데이터의 압축, 전송, 저장에 관한 것을 다루는 분야이다. 이 과정들 중에서 에러를 최소화하고 효율적인 데이터의 압축등을 정보 이론을 이용해 나타낸다. 간단한 예를 보자. 톨게이트에서 차가 한대 지나갈 때 마다 한 컴퓨터에 정보가 저장된다고 하자. 그리고 승용차가 지나가면 a, 트럭이 지나가면 b가 기록되고 a, b 는 각자의 용량을 가진다고 하자. 일반적으로는 승용차가 트럭보다 훨씬 많이 지나갈 것이다. 만일 a의 용량이 크고 b의 용량이 작다면 내 컴퓨터에 하루에 쌓이는 용량은 엄청나게 커지게 될 것이고, 반대로 a가 작고 b가 크다면 이전보다는 용량이 적게 쌓일 것이다. 정보이론은 후자의 경우처럼 효율적인 상황을 추구한다. 신기하게도 기계학습엔 이 정보이론의 관한 것들이 조그맣게 숨어있는데 엔트로피, KL divergence 가 그것이다.
<h2>Entropy, Cross entropy</h2>
엔트로피는 물리에서도 나오는 개념인데 정보이론에서의 엔트로피와 크게 다르지 않다. 물리에서는 에너지의 무질서함의 정도, 정보이론에서는 정보의 무질서함의 정도를 나타내는 개념들이다. 그리고 이 엔트로피는 기계학습에서는 모델의 예측값과 라벨간의 무질서함의 정도를 측정한다. 무질서함이 없으면 예측값과 라벨이 똑같다는 것, 무질서함이 최대값이면 완전히 다르다는 것이다. 무질서함을 측정한다는 것이 좀 이상한데 uncertainty를 측정하는 것이다.

엔트로피는 사실 확률변수이다. 정확히는 분포 p에서의 확률 변수 X이고 $\mathbb{H}(X) , \mathbb{H}(p)$로 표기한다. 그리고 이산 변수에서의 엔트로피는 다음과 같이 정의된다.
$$\mathbb{H}(X) = - \sum ^ {K} _ {k=1} p(X=k) log _ {2} p(X=k)$$

<img class="aligncenter size-full wp-image-850" src="http://dllab.xyz/wp-content/uploads/2018/03/스크린샷-2018-03-24-오후-3.38.47.png" alt="" width="824" height="522" />베르누이 분포에서 엔트로피는 다음과 같은 모습을 보인다. 확률이 0과 1에서 엔트로피가 0인데 베르누이에서 p가 0과 1인 경우 한가지 값만 나오기 때문에 무질서함이 전혀 없을 것이다. p가 0.5일 경우는 동전 던지기와 같은 경우로 이때는 앞면, 뒷면이 반반씩 나오니 결국 무질서함이 최대일 것이다.

그런데 단지 이 엔트로피는 한가지 분포 p에 대한 함수이다. 예측값과 라벨은 분명 서로 다른 분포를 띄고 있으니 엔트로피는 우리에게 맞지 않는다. 두 가지 분포에 대한 함수가 바로 cross entropy이다. Cross entropy는 라벨이 P 분포를 따를 때, Q 분포를 따르는 예측값을 무질서도를 정의할 수 있다. Cross entropy는 다음과 같다.

$$ \mathbb{H}(p,q) = - \sum _ {k} p _ {k} log q _ {k} $$

만일 우리의 예측값이 P분포를 따르고 라벨도 P분포를 따른다면 $\mathbb{H} (p,p)$이고 이것은 $\mathbb{H}(p)$와 같다.
<h2>KL divergence</h2>
KL divergence, Kullback-Leibler divergence는 요즘 특히 베이지언, VAE와 GAN을 보다보니 익숙해지게 되었다. 위 세가지 모델에서 KL은 이해하기에 매우 중요하다. KL divergence는 두 확률 분포의 dissimilarity를 측정한다. 두 분포간의 거리를 측정한다고 봐도 좋다. 정의는 다음과 같다.

$$\mathbb{KL} (p||q) = \sum ^{K} _ {k=1} p_{k} log \frac{p _ {k}}{q _ {k}}$$
위의 로그를 쪼개어 나타내면 다음과 같다.

$$\mathbb{KL} (p||q) = \sum _{k} p _ {k} log p _ {k} - \sum _ {k} p _ {k} log q _ {k} = - \mathbb{H}(p) + \mathbb{H}(p,q)$$

즉 -entropy와 cross entropy의 합이다. 직관적으로 보면 cross entropy에서 entropy를 뺀 것이다. 그리고 entropy인 $\mathbb{H}(p)$는 cross entropy로 나타내면 $\mathbb{H}(p,p)$이다. 그러니 KL은 p 분포의 라벨에서 우리가 q 분포의 예측값을 쓸 때의 엔트로피에서 p 분포인 모델(true model)을 사용했을 때의 엔트로피 값을 뺀 것이다. 우리의 모델이 true model에 가까워 질수록 q는 p에 가까워지면서 KL은 0이 된다. 즉 KL의 최솟값은 0이라는 것이다. 그래서 거리의 개념으로도 볼 수 있다. 이것을 증명하는 정리를 보자.
<h3>Thm Information inequality</h3>
$\mathbb{KL} (p||q) \geq 0$ with equality iff $p=q$
먼저 Jensen’s inequality를 알아야 한다. 어떤 convex function f 가 있을 때, $f(\sum \lambda x) \leq \sum \lambda f(x)$ 라는 것이 jensen’s inequality이다. 이제 본 증명을 보면,
$A = \{x : p(x) &gt; 0 \}$ 이라고 하자.
$- \mathbb{KL} (p||q) = -\sum p(x) log \frac{p(x)}{q(x)} = \sum p(x) log \frac{q(x)}{p(x)}$ log는 concave 함수이니 jensen’s inequality 를 적용할 수 있다. 또 $-\mathbb{KL}$임을 주의하자.
$\leq log \sum p(x) \frac{q(x)}{p(x)} = log \sum q(x) $
$\leq log \sum_{x \in \Omega} q(x) = log 1 = 0$ 확률의 정의에 의해 전체 집합에 대해 확률 q의 합은 1이다.
따라서 $\mathbb{KL} \geq 0$ 이다.
