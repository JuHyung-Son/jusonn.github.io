---
id: 812
title: Multivariate Gaussian 정리
date: 2018-03-20T16:22:23+00:00
author: JuHyung Son
layout: post
permalink: '/multivariate-gaussian-%ec%a0%95%eb%a6%ac/'
image: /wp-content/uploads/2017/11/cropped-nature-2609118-2-250x250.jpg
categories:
  - Deep Learning
  - MATH
---
<h1>Multivariate Gaussian</h1>
<div>나의 경험상, 실제로는 univariate gaussian을 다루기보단 multivariate gaussian을 다뤄야 할 상황이 더 많았다. 하나의 하나의 변수만으로 무얼한다는 건 튜토리얼에서만 있는 일이고 그것이 있다해도 아주 간단한 일이었다. 그런데 multivariate gaussian 은 기본적으로 공부하기 싫다. 식도 훨씬 복잡해지고 차원이 높아지면 눈에 보이지도 않는다. 그러나  실제 데이터는 multivariate 임이 거의 확실하기 때문에 Mutivariate Gaussian 은 꼭 알아야 한다.</div>
<h2>Univariate Gaussian</h2>
<div>변수가 하나인, 그냥 정규분포다. 확률과 통계 책에서 가장 중요한 부분이지 않을까. 정규분포가 가장 많이 쓰이는 이유는 Central limit theorem(CLT), 중심 극한 정리로 알 수 있다. CLT 는 간단히 말해 어떤 분포가 충분한 조건을 만족하면 (보통 데이터가 충분히 많다는) 그 분포는 어떤 분포라도 정규분포에 근사할수 있다는 정리이다. 그리고 정규분포는 평균과 분산만 알면 분포에 관한 모든 정보를 알 수 있는 좋은 분포이다.</div>
<div><a href="http://onlinestatbook.com/stat_sim/sampling_dist/">CLT를 시뮬레이션하는 곳은 이곳</a></div>
<div>그렇기 때문에 정규분포가 많이 쓰이고 일변수 정규분포와 그 pdf 는 다음과 같다.</div>
<div>$$X \sim N( \mu , \sigma ^ {2}  ) , \mu \in \mathbb{R} , \sigma ^ {2} &gt; 0$$</div>
<div>$$f(x) = \frac{1}{\sqrt{2 \pi \sigma ^ {2}} } e ^ {- \frac{1}{2 \sigma ^ {2}} (x - \mu ) ^ {2} } $$</div>
<div>Multivariate gaussian 은 단지 이런 gaussian이 두개 이상 있는 분포다.</div>
<h2>Multivariate Gaussian</h2>
<div>두가지 버전의 Multivariate Gaussian 정의를 소개한다. 전자는 좀 더 일반적인 정의이고 후자는 좀 더 이해가 쉬운 정의이다.</div>
<div></div>
<div>Def)</div>
<div>A random variable $X \in \mathbb{R} ^ {n} $ is multivariate Gaussian.</div>
<div>If any linear combination of it’s components is univariate gaussian.</div>
<div>i.e. $a ^ {T} X = \sum a x$ is univariate gaussian.</div>
<div>multivariate 분포의 모든 각 분포들의 linea combination은 정규분포라는 이야기다. 정규분포들은 연산한 결과는 정규분포이기에 그렇다.</div>
<div></div>
<div>Def)</div>
<div>$X \sim N( \mu , C ) , ~ \mu \in \mathbb{R} ^ {n} , ~C : positive semi definite matrix $ means $X$ is gaussian with $E ( X _ {i} ) = \mu _ {i} $ and $cov (X _ {i} , X _ {j} ) = C _ {ij} $</div>
<div>여기서 C는 Covariance이고 covariance는 positive semi definite라는 특징을 가지고 있다.</div>
<div></div>
<div>Multivariate gaussian 은 2변수 일 때, 그 특징을 그래프로 쉽게 그려볼 수 있다.  그림이 허접하지만 아래가 2 variate gaussian에서 나타나는 형태이다.</div>
<div><img class="aligncenter size-full wp-image-814" src="http://dllab.xyz/wp-content/uploads/2018/03/81470A14-1506-45C5-B2C1-AEAB915904F3.png" alt="" width="1602" height="355" /></div>
<div></div>
<div>다음은 두가지 유용한 정리를 보자.</div>
<h3>Property</h3>
<div>1. $X _ {1} , … , X _ {n}$ are independent with $X _ {i} \sim N( \mu _ {i} , \sigma _ {i} ^ {2} )$ iff $ X = ( X _ {1} , … , X _ {n} ) \sim N ( \mu , C )$ where $ \mu = ( \mu _ {1} , … , \mu _ {n} ) , C = \begin{pmatrix} \sigma _ {1} ^ {2} &amp; ... &amp;0 \\ 0 &amp; \sigma _ {k} ^ {2}  &amp;0 \\ 0 &amp; ... &amp; \sigma _ {n} ^ {2} \end{pmatrix} $</div>
<div></div>
<div></div>
<div>2. If $X \in \mathbb{R} ^ {n} $ is gaussian then, $ X _ {i} , X _ {j} $ are independent iff $Cov(X _ {i} , X _ {j} ) = 0$</div>
<div>둘다 independent에 관한 정리이다. 선형회귀를 할 때, 이 independent는 아주 중요하며 correlation 이란 방법으로 이것을 분석한다.</div>
<div></div>
<div>또 multivariate gaussian의 linear combination 도 multivariate gaussian이다. 정규분포와 매우 비슷하고 행렬을 이용한 표기 방법에만 익숙해지면 된다.</div>
<div></div>
<h3>Property. (Affine Property)</h3>
<div>Any affine transformation of a gaussian is gaussian. That is, if $X \sim N ( \mu , C)$ then $ AX + b \sim N( A \mu + b , ACA ^ {T} ) $</div>
<div></div>
<h2>Plot Multivariate Gaussian</h2>
<div>2개의 변수를 가진 정규분포가 있다고 하자. $\mu = 0$ 일 때, Covariance에 따른 정규분포의 모양을 보자.</div>
<div>이런 모양이다. Multivariate gaussian 은 생각보다 많은걸 할 수 있다. Andrew Ng의 수업에서는 Anormaly detection도 소개하고 등등 생각이 안나는 몇가지를 더 한다.</div>
<div><img class="aligncenter size-full wp-image-815" src="http://dllab.xyz/wp-content/uploads/2018/03/스크린샷-2018-03-19-오후-3.28.20.png" alt="" width="1320" height="1400" /></div>
<div></div>
<h2>Marginal distribution</h2>
<div>반대로 어떤 multivariate gaussian인 분포가 주어졌을 때, 그것으로 부터 marginal distribution을 구할 수있다.</div>
<div>즉, $X = (X _ {1} , X _ {2} ) ^ {T} \in \mathbb{R} ^ {2}$ 라면 $X _ {1} , X _ {2}$는 gaussian 이다. 물론 2개의 변수 뿐만 아니라 n개의 변수가 있는 general한 상황에서도 그렇다.</div>
<div><img class="aligncenter wp-image-817" src="http://dllab.xyz/wp-content/uploads/2018/03/PNG-image.png" alt="" width="653" height="448" /></div>
<div>이것은 증명이 매우 쉬우니 한번 보자.</div>
<div></div>
<div>Pf)</div>
<div>$A = (1, 0)$(projection matrix) , b=0</div>
<div>$X \sim N( \mu , C ) \rightarrow AX+b \sim N( A \mu + b, ACA ^ {T}$</div>
<div>$X$가 Multivariate gaussian(M.G) 이고 그렇다면 이것의 linear combination 형태도 M.G 이다. 결국 $AX = X _ {1}$이므로 $X _ {1} $은 Gaussian이다.</div>
<div></div>
<div>좀 익숙해졌다면 이제 general 한 경우를 봐야한다.</div>
<div></div>
<div>$X \sim N( \mu , C ), ~ a=(1,...,k) , ~b=(k+1,...,n),~ 1  \leq k \leq n $ 라 하자. 그렇다면 $ X = ( X _ {a} , X _ {b} ) ^{T} , X _ {a} = (X _ {1} , ... , X _ {k}) ^ {T} , X _ {b} = ( X _ {k+1}, ... , X _ {n} , \mu = (\mu _ {a} , \mu _ {b}) ^ {T} , C =  \begin {pmatrix} C_{aa} &amp; C_{ab} \\  C_{ba} &amp; C _ {bb} \end {pmatrix} $ 마지막에 $C _ {aa}$ 는 $ x _{1} \sim x _ {k}$ 까지의 covariance matrix다.</div>
<div></div>
<div>또 다음의 정리도 유용하다.</div>
<div></div>
<h3>Prop.</h3>
<div>$X = ( X _ {1} , X _ {2} ) ^ {T} \in \mathbb{R} ^ {2}, Gaussian \rightarrow (X _ {1} | X _ {2} =x _ {2}$ is gaussian.</div>
<div></div>
<div></div>
<div>즉, M.G의 Conditional distribution도 가우시안이라는 것이다. 신기하게 모든 것이 가우시안이다. 이래서 Multivariate Gaussian이 좋은 것이다.</div>
<div></div>
<div>

<img class="wp-image-816" src="http://dllab.xyz/wp-content/uploads/2018/03/PNG-image-2-1024x823.png" alt="" width="464" height="373" />

<img class="wp-image-813 alignnone" src="http://dllab.xyz/wp-content/uploads/2018/03/9SIxb.png" alt="" width="594" height="242" />

</div>
<div></div>
<div></div>
<div></div>
<div></div>
<div></div>
<div></div>
<div></div>
<div>Multivariate Gaussian이 실제로는 어떻게 쓰이는지는 정확히는 모르겠지만, 위의 특징들은 M.G가 굉장히 편하게 쓰일 것 같다는 생각이 들게 한다. 모든 것이 가우시안이고 데이터 자체가 Multivariate gaussian이라면 별다른 가정 없이 데이터를 다룰 수 있을 거 같다는 느낌이다.</div>
