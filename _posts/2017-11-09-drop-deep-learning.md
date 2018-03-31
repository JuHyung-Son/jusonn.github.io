---
id: 252
title: 'Drop out in Deep learning &#8211; paper review'
date: 2017-11-09T00:24:11+00:00
author: JuHyung Son
layout: post
permalink: /drop-deep-learning/
dsq_thread_id:
  - "6270702576"
image: /wp-content/uploads/2017/11/cesar-couto-423267.jpg
categories:
  - Studying
---
<h2>Dropout: A Simple Way to Prevent Neural Networks from Overfitting - Geoffrey Hinton et tal.</h2>
많은 뉴런과 레이어가 있는 인공신경망은 이전에 있었던 기계학습과는 비교할 수 없이 강력한 기계학습 방법이다. 하지만 모든 기계학습이 그렇듯, 인공신경망도 overfitting이라는 난간을 마주친다. 인공신경망의 특성상 overffiting은 아주 빈번하고 항상 일어나는 문제이다. Geoffrey Hinton은 drop out이라는 방법을 소개하여 overfitting을 소개한다.
<h4>Motivation</h4>
Drop out은 생각지 못한 곳에서 아이디어가 나왔는데, 바로 "The role of sex in evolution"이다. 무성 생식과 유성 생식이 drop out의 아이디어이며 두 가지 염색체가 섞이는 유성 생식이 가장 발달한 진화 방법이라는 것에서 착안했다고 한다.
<h4>Drop out</h4>
drop out의 핵심 아이디어는 학습 과정 중에 신경망의 뉴런들을 임의로 꺼버리는 것이다. 이 방법은 overfitting이 일어나는 걸 효과적으로 막아준다. 학습 중에 drop out은 기하급수적으로 많은 서로 다른 'thinned' 네트워크를 모은다. 테스트 과정에서는 모든 뉴런, 즉, 'un-thinned' 네트워크를 사용함으로 기하급수적으로 많은 'thinned' 네트워크의 평균에 근사값을 가지는 효과를 가진다. 실제로 이건 overfitting을 엄청나게 줄이면서 다른 효과들도 보여준다.컴퓨팅 능력에 제한이 없다면 예측을 하는 가장 좋은 방법은 가능한 모든 파라미터를 사용하는 것이지만 이것은 역시 불가능하다. 제한된 컴퓨팅으로 우리가 하고자 하는 것은 단지 가장 좋은 방법과 비슷한 퍼포먼스를 보이는 방법을 찾는 것이다.

컴퓨팅 능력에 제한이 없다면 예측을 하는 가장 좋은 방법은 가능한 모든 파라미터를 사용하는 것이지만 이것은 역시 불가능하다. 만일 이것이 가능하다면 이때의 에러를 The bayesian gold standard라고 한다. 제한된 컴퓨팅으로 우리가 하고자 하는 것은 단지 the bayesian gold standard와 비슷한 에러를 가지는 방법을 찾는 것이다. 이전에 제안되었던 방법은 기하급수적으로 많은 학습된  네트워크들의 예측값의 voting을 하는 방법이다. 기계학습에서 많이 쓰이는 ensemble 기법이다. 하지만, 인공신경망에서는 하나하나의 네트워크의 하이퍼 파라미터를 찾아 최적화하는 것부터가 힘들 뿐더러 각각의 네트워크를 학습시키는데에도 큰 비용이 들고 필요한 데이터의 양도 어마어마하다.

Drop out은 이러한 인공신경망에서의 ensemble 문제를 해결해준다. drop out은 overfitting을 막으면서 기하급수적으로 많은 인공신경망을 효과적으로 ensemble하는 효과도 가진다. drop out은 말그대로 한 뉴런을 끈다(dropping out)고 하는 것으로 일시적으로 뉴런을 네트워크에서 없애는 것이다. 어느 뉴런이 꺼질지는 랜덤으로 보통 drop out probability라고 하며 0.5~1의 값을 주고 이 값 역시 하이퍼 파라미터이다. 학습이 끝난 후 테스트 과정에서는 drop out probability를 1로 준다. Drop out을 하지 않는 것이다. 그리고 여기서 ensemble과 비슷한 효과가 일어나는 것이다. Drop out은 feed forward 네트워크 뿐만 아니라 RBM, autoencoder 등 여러 네트워크에 사용할 수 있다.

<a href="http://jmlr.org/papers/v15/srivastava14a.html">원문보기</a>

&nbsp;
