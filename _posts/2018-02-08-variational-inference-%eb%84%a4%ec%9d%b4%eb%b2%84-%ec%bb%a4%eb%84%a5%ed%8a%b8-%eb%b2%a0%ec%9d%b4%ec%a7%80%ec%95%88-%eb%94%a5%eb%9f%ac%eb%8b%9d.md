---
id: 678
title: 'Variational Inference,  베이지안 딥러닝'
date: 2018-02-08T17:51:17+00:00
author: JuHyung Son
layout: post
dsq_thread_id:
  - "6466000779"
tags:
  - variational inference
  - bayesian
categories:
  - MATH
---
네이버 커넥트 재단에서 최성준 박사님이 강의한 베이지안 딥러닝의 내용입니다.

<h2>Variational Inference</h2>
개인적으로 Variational inference 라는 단어를 들어 본 지는 사실 꽤 되었습니다. 이게 뭘까 하면서 한 번 본 적은 있지만, 난무하는 수식에 그냥 하던거나 하자는 생각으로 덮었던 게 두번 정도 입니다. 그래서 사실 variational inference가 뭔지도 모르고 이 variational이 variance와 관련 있는건가 하는 궁금증도 들고 했습니다. 어쨋든 이 둘은 다른 이야기입니다.

강의에서는 4개의 논문을 소개하며 결국 4개 모두 variational inference를 보여주는 것을 보여줍니다. 모두 베이지언의 토대를 이룬 논문들로 심심하면 읽어볼만 할 거 같은데 사실 읽어볼 거 같진 않아요. 첫 논문은 variational inference의 가장 첫 논문이라고 여겨지는 것들인데 지금의 것과는 많이 다르고 variational 이라는 용어 자체도 나오지 않습니다. 분야도 생소한 분야라 잘 모르겠고요. 전 그래서 이번 강의에서 가장 좋았던 내용은 그래서 variational inference 뭔지를 알아가는 과정이었습니다.

<h2>Bayesian</h2>

이야기는 베이지언에서 시작됩니다. 대부분의 베이지언 문제들에서 posterior를 구하고 싶지만 사실상 그게 되질 않습니다. 그래서 여러 방법들이 나오는데 variational inference가 그 중의 한 방법이죠. 우리가 구하고자하는 posterior는 다음과 같습니다.

$$ P(Z|X) = \frac {P(Z,X)} {\int _ {z} P(Z _ {z} | X)} $$

보통 대부분 분모에 있는 적분을 구하는건 불가능합니다. 먼저 이걸 쉽게 근사하는 방법은 Markov chain Monte Carlo(MCMC)로 posterior 분포에서 샘플을 가져와 계산을 한 후 그것의 평균을 내는 방법입니다. Metropolis-Hastings algorithm 혹은 Gibbs sampling이 MCMC의 예 중 하나 입니다. 하지만 이 방법은 파라미터가 많으면 수렴이 매우 늦어진다는 단점이 있습니다. 그래서 사용하는 것이 variational inference 입니다.

<h2>variational distribution</h2>

$$ P(Z|X) \approx Q(Z|V) = \prod _ {i} Q(Z _ {i} | V _ {i}) $$

Variational inference에서는 variational distribution이라는 것을 만듭니다. 이건 posterior에 근사하는 분포입니다. Gaussian으로 근사하는 거 같은데 정확히는 아직은 모르겠네요. 다른 것도 되지만 gaussian이 가장 쉽고 좋기 때문에 사용한 거 아닐까요. variational distribution은 다음과 같이 나타냅니다. X는 데이터이고 Z는 근사하는데 사용되는 변수입니다.
이제 우리의 목표는 true posterior에 근접한 variational distribution을 만드는 겁니다. 이 두 분포의 비슷함의 정도는 Kullback-Leibler Divergence 라는 간단히 말해 분포간의 거리를 측정하는 방법으로 계산합니다. 이제 $ KL(Q || P) $를 최소화하는 것이 목표죠. 그런데 이 $ KL(Q || P) $를 구하는 것도 불가능합니다. 다른 방법을 써야하죠.

그런데 $p(Z|X)$는 다음과 같이 표현이 가능합니다.
<div align="center">
<img class="wp-image-680 size-full" src="/wp-content/uploads/2018/02/스크린샷-2018-02-08-오후-5.36.08.png" alt="" width="766" height="322" /> </div> notation이 조금 다지만, 이해는 가능할 겁니다.

이 맨 마지막 식을 Evidence Lower Bound, ELBO라고 합니다. 이 ELBO를 Variational free energy로 정의합니다.
<div align="center"> <img src="/wp-content/uploads/2018/02/스크린샷-2018-02-08-오후-5.37.36.png" alt="" width="580" height="170" /> </div>
<div>
그리고 아까 있던 $KL(Q||P)$ 는 다음과 같이 표현할 수 있습니다.</div>
<div>
신기하게 Variational free energy와 posterior의 로그 합이 $KL(Q||P)$ 입니다. 그래서 밑 그림과 같은 표현이 가능합니다.
</div>
<div align="center"><img src="/wp-content/uploads/2018/02/스크린샷-2018-02-08-오후-5.38.53.png" alt="" width="784" height="211" /></div>
<div>
그래서 posterior도 못구하고 $KL(Q||P)$ 도 못구하니 아래 Variational free energy를 최대로 만들어 Variational distribution 을 posterior에 근사시키는 것이 variational inference 입니다.
</div>
<div align="center"><img class="wp-image-683 size-full" src="/wp-content/uploads/2018/02/스크린샷-2018-02-08-오후-5.40.06.png" alt="" width="588" height="344" /></div>

사실 이 모든 식을 유도해 나가는 과정은 몇 개 안된다고 하시지만 굉장히 복잡해 보입니다. Variational Inference에 관한 강의나 책을 보아도 난무하는 수식에 정신을 못차리겠네요.
<h4>참조</h4>
최성준 박사님 강의노트

<a href="https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf">Variational inference - David M Blei</a>
