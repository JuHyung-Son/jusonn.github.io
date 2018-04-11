---
id: 726
title: 쉽게 본 Measure Theory ~ Probability
date: 2018-02-24T02:53:17+00:00
author: JuHyung Son
layout: post
tags:
   - Measure theory
   - probability
categories:
  - MATH
---

Youtube의 mathematical monk 채널을 구독 중입니다.
<h1>Measure Theory</h1>
<h2>Banach- Tarski Paradox</h2>
<div align="center">
<img class="aligncenter size-full wp-image-730" src="/wp-content/uploads/2018/02/445px-Banach-Tarski_Paradox.png" alt="" width="445" height="100" /> </div>

먼저 Measure가 무엇인지 직관적인 이해를 돕기 위해 바나흐-탈스키의 역설을 봅니다. 이 역설은 3차원 상의 공을 유한 개의 조각으로 자른 다음 늘리거나 변형하지 않고 재조합만 하면 원래 공과 같은 부피를 갖는 공 두 개를 만들 수 있다는 정리입니다. 이 말도 안되는 상황을 증명하기 위해 바나흐의 탈스키는 체르멜로-프렝켈 집합론에서 선택 공리를 추가하여 증명했다고 합니다. (ZFC set 이라고 한다.) 물론 이건 역설이니 틀렸음을 증명할 수 있어야 하는데 수학자들이 이것을 증명하는데 꽤 애를 먹었다고 합니다. 이것을 틀렸음을 증명하는데는 두가지 방법이 있습니다. 먼저 선택 공리(axiom of choice)라는 것을 제외하거나 혹은 공 조각들이 non-measurable set이라는 것을 증명하면 됩니다. non-measurable 하다는 것은 부피나 크기 같은 어떤 값을 측정할 수 없다는 얘기입니다. 조각들이 non-measurable이니 Borel set이 아니게 되고 역설인 것이 증명됩니다. 실제로 우리가 보는 보는 집합, 함수는 measurable 해서 non-measurable 하다는 것이 잘 와닿지 않습니다. 예로 어떤 것이 있는지도 잘 생각이 나질 않고요. 여전히 measure에 대해서는 잘 모르겠습니다. 어쨋든 이 부분에 대해 더 궁금하면 아래의 논문을 보면 됩니다.

<a href="https://www.math.hmc.edu/~su/papers.dir/banachtarski.pdf">The Banach-Tarski Paradox 보기</a>
<h2>$\sigma$ - algebra</h2>
Measure가 뭔지 알아가기 위해 먼저 $\sigma$ - algebra를 정의합니다.
어떤 집합 $\Omega$가 있다면 $\Omega$에서 $\sigma$ - algebra는 $\Omega$의 power set에 공집합이 아닌 부분집합 A이고 다음 두 조건을 만족합니다.
<ol>
 	<li>Closed under complements. $E \in A \rightarrow E ^ {c} \in A$ 원소에 대해 닫혀있다.</li>
 	<li>Closed under countable unions. $E _ {1} , E _ {2} …. \in A \rightarrow \bigcup E _ {i} \in A$ 셀 수 있는 합연산에 대해 닫혀 있다. </li>
</ol>
<h4>Ex</h4>
<ol>
 	<li>$A = \{ \varnothing , \Omega \}$ 가장 간단한 예제로 위의 모든 조건을 만족합니다.</li>
 	<li>$A = \{ \varnothing, E, E ^ {c} , \Omega \} $</li>
</ol>
<h2>Measure</h2>
이제 Measure의 정의를 봅니다. Measure는 $ \sigma$ - algebra A를 실수로 보내는 함수이고 다음의 두 조건을 만족해야 합니다.
<ol>
 	<li>공집합의 measure 는 0이다. 그러므로 measure는 항상 0 이상이다.</li>
 	<li>A의 부분집합 $E _ {i}$ 의 합집합의 measure는 $E _ {i}$의 measure의 합과 같다.</li>
</ol>
여기서 조건 2는 countable additivity 라고 불립니다. 위의 정의를 수학적으로 표현해 좀 더 간결하게 나타내면 다음과 같습니다.

A **measure** $\mu$ on $\Omega$ with $\sigma$ - algebra A is a function $\mu : A \rightarrow [0, \infty ]$ such that <ol>
 	<li>$\mu ( \varnothing) = 0$</li>
 	<li>$\mu (\bigcup E _ {i}) = \sum \mu ( E _ {i} ) $ for any $E _ {i} \in A$ of pairwise disjoint.</li>
</ol>

이렇게 measure가 정의되었습니다. Measure는 쉽게는 어떤 집합의 부분 집합을 한 숫자로 보내는 함수입니다. Size, volume, weight 같은 것들이 measure라고 할 수 있습니다. 확률이 이 measure로 부터 정의되는데, 확률이 [0, 1]의 값을 가진다는 걸 알면 [0 무한]의 값을 가진 measure를 [0, 1]로 scaling하면 비슷하다는 걸 알 수 있다.
<h2>Probability</h2>

**Probability measure** is a measure P such that $P ( \Omega) = 1$

그리고 이 Probability measure는 통계학에서 배우는 확률의 조건들을 딱 들어 맞습니다. 당연히 probability measure가 정의되고 확률이 정의되었기 때문입니다. 확률은 kolmogrov에 의해 수학적으로 엄밀하게 정의되었다고 합니다. 이후 확률은 도박사들이 하는 일종의 장난에서 학문의 반열로 올랐다는 카더라도 있습니다.
<h4>Ex)</h4>
위의 이론들을 가지고 흔히 배우는 확률을 기술할 수도 있습니다.
<ol>
 	<li>$\Omega = \{ 1, 2, …., n \} , ~ A = 2 ^ {\Omega} $이고 $P( \{ k \}) = P(k) = \frac {1} {n} \forall k \in \Omega $이라면 이 확률은 uniform 분포를 따른다.</li>
 	<li>$\Omega = \{1,2,… \}, ~ A = 2 ^ {\Omega} $ 이고 $P(k) =$ 동전 던질때 앞면이 나올 확률이라고 하면, 확률은 $a(1-a) ^ {k-1}$이다. 이 경우 확률은 geometric 분포를 따르게 된다.</li>
</ol>

<h2>Basic property of Measure</h2>

이 기본 정리는 여러 정리들을 증명하는데 쓰이는 정리입니다
$ ( \Omega , A , \mu) $ 가 measure space라고 하자. 그러면 다음 3가지 조건이 만족한다.<ol>
 	<li>monotonicity: $E,F \in A$이고 $E \subset F$ 이면 $\mu (E) \leq \mu (F)$ 이다.</li>
 	<li>subadditivity: $E _ {1}, E _ {2}, … \in A$이면 $\mu ( \bigcup E) \leq \sum \mu (E)$</li>
 	<li>continuity from below: $E _ {1}, E _ {2}, … \in A$ 이고 $E _ {1} \subset E _ {2} \subset …$ 이면 $\mu ( \bigcup E) = lim \mu (E)$</li>
 	<li>continuity from above: $E _ {1}, E _ {2}, … \in A$ 이고 $E _ {1} \supset E _ {2} \supset … $ 이고 $ \mu (E _ {1} ) < \infty $ 이면 $\mu ( \bigcap E) = lim \mu(E)$ 이다.</li>
</ol>

어려워 보이지만 사실 읽어보면 아주 당연한 정리이고 measure에 대해 이해할 것 같은 느낌이 오기도 합니다.
<h2>Facts</h2>
다음 6가지 식은 아주 기본적인, 중요한 확률의 사실이고 확률 수업이나 책 초반에 나오는 기본 정리이고 끝까지 나오는 정리입니다.

Let $( \Omega , A, \epsilon) $ be probability measure space with $E, ~ F, ~ E _ {i} \in A$ <ol>
 	<li> $P(E \cup F) = P(E) + P(F) $ if $E \cap F = \varnothing$</li>
 	<li>$ P(E \cup F) = P(E) + P(F) - P( E \cap F) $</li>
 	<li>$ P(E) = 1 - P(E ^ {c}) $</li>
 	<li>$ P(E \cap F ^ {c} ) = P(E) - P( E \cap F ) $</li>
 	<li>$ P( \bigcap ^ {n} _ {i=1} E _ {i}) = \sum P ( E _ {i} - \sum _ {i < j} P ( E _ {i} \cap F _ {j}) +....+ (-1) ^ {n+1} P (E _ {1} \cap .... \cap E _ {n} ) ) $</li>
 	<li>$P(\bigcup E _ {i}) \leq \sum P (E _ {i} ) and P (\bigcup E _ {i} ) \leq \sum P ( E _ {i} ) $</li>
</ol>

<h2>Measures on $\mathbb{R}$ , $Borel ( \mathbb{R} )$</h2>
맨 처음에 잠깐 나왔던 Borel measure는 모든 열린 집합에서 정의된 measure 이고 열린 집합들로부터 연산하여 만들 수 있는 집합입니다. 일반적으로 $ \mathbb{R} $ , $Borel ( \mathbb{R} )$ 는 단조함수를 이용해 특별한 형태로 바꿀 수 있습니다. 그리고 probability borel measure에서는 이것이 더 간단한 형태를 갖습니다. 여기서 단조함수란 수업에서 배운 CDF입니다. Borel measure와 CDF의 Theorem을 각각 봅시다.
<div></div>

A **Borel measure** on $\mathbb{R}$ is a measure on $ ( \mathbb{R} , B( \mathbb{R} ) )$

<h4>Thm)</h4>
<ol>
 	<li>If F is a CDF, then there is a unique borel probability measure on $ \mathbb{R} $ such that $ P( ( - \infty , x]) = F(x), ~ \forall x \in \mathbb{R} $</li>
 	<li>If P is a Borel probability measure on $ \mathbb{R} $ then there is a unique CDF $F$ such that $F(x) = P (( - \infty , x]) , ~ \forall x \in \mathbb{R}$</li>
</ol>
이 theorem에 의하면 CDF 와 Borel probability measure는 equivalent 즉, 동치관계입니다. 그래서 CDF만 알면 그것의 PDF도 구할 수 있기 때문에 CDF가 중요하다고 하는 것입니다.
