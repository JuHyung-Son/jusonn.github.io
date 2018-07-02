---
id: 454
title: Maximum likelihood의 통계적 접근
date: 2017-11-27T17:51:42+00:00
author: JuHyung Son
layout: post
categories:
  - MATH
tags:
  - statictics
  - mle
---
<h2>Maximum likelihood</h2>

Maximum Likelihood Methods는 통계에서 파라미터를 추정하는 통계 방법입니다. Likehood를 최대화하는 파라미터를 찾아 그것을 $\hat{\theta}$로 추정하는 거죠. mle는 통계학에서 만들어졌지만 사실 거의 모든 분야에서 볼 수 있을 정도로 많이 쓰이는 방법입니다. 기계학습, 딥러닝에서도 물론 mle를 통해 파라미터를 추정합니다.

Mle는 통계의 아버지라고 불리는 Fisher에 의해 제안되었습니다. 그 이전에서 Carl Friedrich Gauss, Pierre-Simon Laplace, Thorvald N. Thiele, and Francis Ysidro Edgeworth 에 의해 쓰이던 방법이기는 하지만 fisher가 뭔가 공식적으로 발표를 했다고 합니다. 그리고 mle가 가장 최종적으로 증명된 것은 Samuel S. Wilk가 1938년에 하였습니다.

<div align='center'> <img class="size-medium" src="https://upload.wikimedia.org/wikipedia/commons/a/aa/Youngronaldfisher2.JPG" width="291" height="408" /></div>
<div align = 'center'>Fisher</div>

<h2>Intuitive</h2>

mle을 직관적으로 봅시다. mle는 likelihood라는 함수를 최대화하는 파라미터를 선택하는데, 이것은 모델이 관찰된 데이터와 얼마나 일치하느냐를 최대화하는 겁니다. likelihood function은 다음과 같습니다. 

$ f(x; \theta) $: pdf이고 $ X _ {1},…..X _ {n} $가 iid하다면, $ L( theta ; x) = \prod _ {i=1} f(x _ {i}; \theta) $

$$L( \theta ; x) = \prod n f(x _ {i} ; \theta), ~ \theta \in \Omega$$

하지만 보통 이 식 그대로를 사용하진 않고 계산의 편의를 위해 log를 씌운 형태가 주로 쓰입니다.

$$l( \theta ) = log L ( \theta ) = \sum _ {i=1} n log f( x_ {i} ; \theta ), \theta \in \Omega$$

log 함수는 concave 함수이기 때문에 미분을 하여 0이 되는 점이 최대이고 즉, $\theta $가 $\hat{\theta}$이 됩니다.

그런데 왜 likelihood를 최대로 만드는 파라미터가 좋은 파라미터가 될까요? likelihood가 높을수록 추정 파라미터가 실제 파라미터에 가깝다고 하였고 그래서 likelihood라는 이름이 붙었습니다. 우리말로는 가능도 함수 라고 합니다. 이번에는 likelihood가 왜 그런 성질을 가졌는지 봅니다.

<h2>Theorem</h2>

$ \theta _ {0}$이 실제 파라미터이고 , regularity conditions 하에서 다음이 성립합니다. (이 regularity condition은 여기서 다루진 않아요.) 

$$lim _ { n \rightarrow \infty} P _ {\theta _ {0}}[L( \theta , X)] =1, ~ for ~all~ \theta \neq \theta _ {0}$ $\theta _ {0}$$

의 likelihood가 다른 $\theta$의 likelihood보다 높을 확률이 1이라는 겁니다. 즉, 가장 크다는 얘기죠. 그러면 likelihood가 높을수록 실제 파라미터에 근접하니 mle는 꽤 괜찮아 보이는 접근 방법입니다. 이 정리를 수식으로 확인해보려면 다음처럼 식을 변형합니다.

$$ln \sum _ {i=1} log f(X _ {i} ; \theta) f(X _ {i} ; \theta _ {0} ) < 0$$

다음, Law of Large Numbers와 Jensen's inequality에 의해 다음으로 확률 수렴합니다.

$$ ln \sum _ {i =1} log f(X _ {i} ; \theta) f(X _ {i} ; \theta _ {0}) \rightarrow P( E _ {\theta _ {0}}[ log f(X _ {1} ; \theta) f(X _ {i} ; \theta _ {0} ) ] < log E _ {\theta _ {0}} ( f(X _ {1} ; \theta) f(X _ {1} ; \theta _ {0}) ) )$$

여기서 

$$E _ {\theta _ 0} ( f(X _ {1} ; \theta) f(X _ {1} ; \theta _ {0})) = \int f(x; \theta) f(x; \theta _ {0})f(x; \theta _ {0}) dx = 1$$

따라서 $log1=0$이므로 위의 식은 성립합니다.

이 정리에 의하면 likelihood는 실제 파라미터인 $\theta _ {0}$에서 최대화합니다. 이런 이유로 파라미터를 추정할 때, likelihood를 최대화하는 파라미터를 선택하는 것이 mle의 기본 아이디어입니다.

<a href="https://onlinecourses.science.psu.edu/stat414/node/191">PenState mle 강의자료 보기</a>

<h2>Properties of MLE</h2>

Mle의 가장 좋은 점은 mle가 가장 좋은 estimator로 보이기 때문입니다. 또 어떤 적당한 조건에서 mle는 consistency를 만족하기도 하죠. 즉, 데이터가 무한히 증가하면 mle는 실제 파라미터에 수렴한다는 이야기입니다.  그 어떤 적당한 조건은 다음과 같습니다.
<ol>
 	<li>실제 데이터 분포는 모델의 분포 안에 있어야한다.</li>
 	<li>실제 데이터의 분포는 단 하나의 파라미터에만 대응해야 한다.</li>
</ol>

