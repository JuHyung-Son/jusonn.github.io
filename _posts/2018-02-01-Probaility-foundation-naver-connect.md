---
id: 653
title: 집합에서 부터 정의하는 확률 , 네이버 커넥트 재단 Bayesian deep learning
date: 2018-02-01T14:09:06+00:00
author: JuHyung Son
layout: post
categories:
  - MATH
tags:
  - probability
---
네이버 커넥트 재단에서 진행한 베이지언 딥러닝의 최성준님 강의를 듣고 인상적인 부분에 관한 내용입니다.
<h2>Probability</h2>
거의 모든 사람들은 초등학교, 중학교를 거치며, 혹은 그냥 살면서 확률에 대한 개념을 습득합니다 저도 그렇고요. 동전과 주사위만 있으면 되는게 확률이니까요. 고등학교에 진학하면 이제 확률을 좀 더 배우지만, 제 기억으론 확률을 설명하라면 설명하지 못했을 거 같습니다. 단지 어떤 사건이 일어나는 것에 대한 믿음의 정도라고 하지 않았을까요. 그 다음, 대학교에서 확률, 수리 통계 수업을 들으면 이제 확률이 뭔지, 그 정의에 대해 이야기 할 수 있게 됩니다. Hogg의 수리통계학 책에는 확률이 다음과 같이 정의되어 있습니다.

Let $C$ be a sample space and let $B$ be the set of events. Let $P$ be a real valued function defined on $B$. Then $P$ is a probability set funcion if $P$ satisfies the following three conditions:
 	<li>$P(C) \geq 0$, for all $C \in B$</li>
 	<li>$P(C) = 1$</li>
 	<li>If $ \{C _ {n} \} $ is a sequence of events in $B$ and $C_{m} \cap C _ {n} = \varnothing $ for all $ m \neq n$, then $$P( \bigcup C _ {n}) = \sum P(C _ {n} )$$</li>

아마 이 정도가 학부 수준에서 배우는 확률일 것입니다. 그리고 대학원에서 확률론을 배우면 정말 0에서 시작하여 확률을 정의하게 된다고 교수님께서 그러셨지요. 이번 강의에서는 이 과정을 아주 쉽고 필요한 것들만 딱 강의 해주셨습니다. 확률에 관심도 많았지만, 확률론은 건드릴 엄두가 안났던 저에겐 아주 좋은 강의였네요.
<h2>Process</h2>
확률은 다음의 과정을 거쳐 정의됩니다.

Set - set function - $\sigma - field \beta$ - measure - probability
쉬운듯 안쉬운 과정 같네요.

Set theory 에선 집합을 정의하고 countable, uncountable, denumerable 등을 정의한 후 function 혹은 mapping을 정의합니다. 이후 measure라는 것을 정의합니다. 학교에서 배우지 않아 약간 생소한 개념이었는데 다음과 같습니다.

> Given a universal set U, a measure assigns a nonnegative real number to each subset of U.

U의 부분 집합에 어떤 음수가 아닌 숫자를 주는 것이라고 한다. 왜 음수여야만 하는지는 잘 모르겠지만 정의니까 그렇다고 합니다. 다음 아주 생소한 개념인 $ \sigma - field$가 등장합니다.

$ \sigma - field \beta$ 란 다음 세개의 조건을 만족하는 U의 부분 집합의 모음입니다.
 	<li>$\varnothing \in \beta$ 공집합이 포합되어야 한다.</li>
 	<li>$ B \in \beta \rightarrow B ^ {c} \in \beta $ 여집합도 포함되어야 한다.</li>
 	<li>$ B _ {i} \in \beta \rightarrow \bigcup B _ {i} \in \beta $ 합집합들도 포합되어야 한다.</li>

$ \sigma - field \beta$ 는 그림으로 그려보면서 생각하면 이해가 좀 더 수월해집니다. $ \sigma - field$는 measure를 정의하기 위해 만들어졌다고 합니다. 즉 원소가 $ \sigma - field$ 에 있지 않으면 그것은 measure 될 수 없습니다. 근데 우리가 지금 껏 배운 수학은 모두 $ \sigma - field$ 안에 있기 때문에 이것이 아닌 것을 생각하기도 힘듭니다. 다음 measure가 무엇인지 보면
>A measure $\mu$ defined on a measurable space $(U, \beta)$ is a set function $\mu : \beta \rightarrow [0, \infty]$ such that,
 	<li>$\mu ( \varnothing) = 0$</li>
 	<li>For disjoint $B _ {i}$ and $B _ {j} \rightarrow \mu (\bigcup B _ {i} ) = \sum \mu (B _ {i})$</li>

굉장히 어려워 보이지만, 읽어보면 직관적으로 당연하게 느껴지는 정의입니다. 확률은 이것으로 정의되는데 먼저 간단하게 생각하면, 확률은 $\mu (U) = 1$ 로 scaling 된 measure입니다. 이제 확률이 나올 차례입니다. random experiment, outcomes, sample point, sample space의 개념을 안다고 하고 바로 확률의 정의를 봅시다.

> P defined on a measurable space $( \Omega, A )$ is a set function $P : A \rightarrow [0,1]$ such that,
 	<li>$P ( \varnothing ) = 0$</li>
 	<li>$P(A) \geq 0 , \forall A \subseteq \Omega $</li>
 	<li>For disjoint sets $A _ {i}$ and $ A _ {j} \rightarrow P( \bigcup A _ {i} ) = \sum P(A _ {i}) $</li>
 	<li>$P( \Omega ) = 1$</li>

마침내 이렇게 확률이 정의됩니다. 수리통계학 수업까지 들었던 내가 확률을 참 모호하게 알았구나 하는 생각이 들었습니다. 또 공부를 어디까지 해야 하는지 혼란이 오기도 하고 뭐 그렇네요.

모든 내용은 네이버 커넥트에서 하는 Bayesian deep learning의 수업에서 가져왔습니다.
