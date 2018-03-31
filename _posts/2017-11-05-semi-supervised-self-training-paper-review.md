---
id: 175
title: 'Semi-supervised self-training &#8211; Paper review'
date: 2017-11-05T21:30:01+00:00
author: JuHyung Son
layout: post
permalink: /semi-supervised-self-training-paper-review/
dsq_thread_id:
  - "6264423455"
image: /wp-content/uploads/2017/11/R1920x0-7-250x250.png
categories:
  - Studying
---
<h1>Semi-supervised self-training for decision tree classifiers</h1>
<h1>Afar Tanha et tal.</h1>
이번에 본 논문은 semi-supervised에 관한 논문이다. 먼저 semi-supervised learning란 라벨이 되어있는 데이터와 라벨이 없는 데이터 둘다 쓰는 것이다. 그래서 semi-supervised self-training은 이러한 데이터를 가지고 자기자신을 학습시키는 학습을 의미한다. 자기 자신을 학습시킨다는 말은 라벨이 없는 데이터에 라벨을 주는 과정을 뜻한다.

이런 학습을 해야할 상황은 꽤 자주 접할 수 있다. 왜냐하면 특정 분야가 아니라면 라벨이 되어있는 데이터를 얻기란 어렵다. 비용이 들고 시간이 든다. 또 그것을 검수할 사람까지 필요한 상황이 있기 때문이다. 주로 이런 상황에서 self-training이 효율적으로 쓰인다.

적은 수의 라벨된 데이터와 상대적으로 더 많은 수의 라벨되지 않은 데이터가 있다면 Semi-Supervised self training은 라벨되지 않은 데이터에 라벨을 붙이는 일을 한다. 물론 예측을 하는 것처럼 한번에 모든 데이터에 라벨을 붙이는 것은 아니다. 가장 확신을 가지고 라벨을 줄 수 있는 데이터만 라벨을 주는 방식이다. 논문에서는 보듯이 Decision tree classifier를 가지고 학습을 하였다. Decision Tree(DT)는 가장 많이 쓰이는 분류 알고리즘의 하나로 파라미터 변환이 필요하지 않고 어떠한 가정도 필요치 않다. 하지만 DT는 좋은 확률적 추정치는 주지 못한다. 논문에서는 이 해결책으로 변형된 DT를 제안한다. 제안 모델은 C4.4, NBTree, Grafted DT, Ensemble이 있다. 논문에서는 이 부분이 꽤 자세히 나와있다. 전체적인 과정은 다음과 같다.

<img class="aligncenter wp-image-177 size-large" src="http://dllab.xyz/wp-content/uploads/2017/11/R1920x0-7-1024x621.png" alt="" width="680" height="412" />
<ol>
 	<li>라벨된 데이터를 학습한다.</li>
 	<li>높은 가능성을 가진 데이터에 pseudo label을 붙인다.</li>
 	<li>2 에서 얻은 데이터를 합하여 다시 학습한다.</li>
 	<li>반복한다.</li>
</ol>
이 학습을 평가하는 방법으로 논문은 Friedman’s test를 사용했다. 이 방식은 두 평균의 차이가 유의미한지를 보여준다. 개인적으로는 이런 self-training 방식과 단지 prediction에 차이가 궁금하다. 방식이 약간 다르지만 self-training은 기본적으로 prediction을 이용하는 것 같다. self-training의 라벨링을 하는 조심성? 때문인지 supervised learning보다 더 좋은 결과를 보여준다.

<img class="aligncenter wp-image-176 size-large" src="http://dllab.xyz/wp-content/uploads/2017/11/R1920x0-8-1024x691.png" alt="" width="680" height="459" />

이 self-training 기법은 개인적으로는 아주 재미있다.

이 논문에서는 수치적인 데이터를 사용하였지만 사실 이러한 데이터에서 self-training을 쓸지는 의문이다.  오늘날 데이터는 끊임없이 생성되기 때문에 굳이 이 방법을 써야할까? 라는 생각이다. 저번에 들은 세미나에서는 이미지 쪽에 이 semi-supervised self-training을 시도해보는 것을 보았다. 이미지에 라벨을 주는 건 아직 사람이 일일히 해야하는 고된 작업이고 어쩌면 불가능할지도 모르기 때문이다. 또 GAN으로 생성되는 데이터와 self-training이 합쳐서 연구를 하는 곳이 꽤 많이 되었다.

&nbsp;

<a href="https://www.infona.pl/resource/bwmeta1.element.springer-doi-10_1007-S13042-015-0328-7">원문 보기</a>
