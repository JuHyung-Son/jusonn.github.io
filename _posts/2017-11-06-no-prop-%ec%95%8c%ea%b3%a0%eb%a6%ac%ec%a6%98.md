---
id: 228
title: No Prop 알고리즘
date: 2017-11-06T18:13:22+00:00
author: JuHyung Son
layout: post
permalink: '/no-prop-%ec%95%8c%ea%b3%a0%eb%a6%ac%ec%a6%98/'
dsq_thread_id:
  - "6265755109"
image: /wp-content/uploads/2017/11/15ZuLCsB1KXEPgHu-qJ8WxQ-250x250.png
categories:
  - Studying
---
<h1>The No Prop algorithm: A new learning algorithm for multilayer neural networks</h1>
<h3>Bernard Widrow∗, Aaron Greenblatt, Youngsik Kim, Dookun Park</h3>
요즘 다시 또 각광 받는 인공신경망은 사실 이전에도 몇번 각광을 받아왔다. 하지만 그때마다 풀지 못했던 문제를 만나 역사 속으로 사라지곤 했다. 지금 다시 대세된 인공신경망은 딥러닝이라 불리고 이전에 풀지 못했던 것을 해결했기에 다시 대세가 될 수 있었다. 암흑기의 인공신경망을 다시 꺼낸 것이 1989년 Yann Lecun et tal. 의 Back Propagation 알고리즘이다. MNIST 또한 Back-Prop와 함께 세상에 나왔다. 현재 딥러닝이라고 하면 당연하게도 Back-prop에 기초한 방법이다. 하지만 그렇다고 모든 딥러닝이 Back-prop를 써야하는 건 아니다. 단지 지금 대세일 뿐. 왜 그런가 하면 인공신경망은 정말 "인공"이다. 뇌를 본 뜬 것이지만 뉴런의 작동 방식에서 아이디어만 따왔지 뇌를 정확하게 구현한 것은 아니다. 그리고 뇌는 실제로 어떻게 작동하는지도 모른다. 그래서 Back-Prop이라는 하나의 방법으로 우리만의 신경망을 작동시키는 것이다. 즉, 다른 방법이 얼마든지 나올 수 있고 이 논문이 그 다른 방법 중 하나인 No Prop 알고리즘이다.

논문에 따르면 No prop 알고리즘은 더 사용하기 간단하고 더 빨리 수렴한다. No prop의 아이디어는 back prop에서 가져왔는데 작동 방식과 결과가 거의 똑같아 back prop의 변형으로 보아도 된다.  이것이 No prop의 장점인데 더 간단하고 학습이 빠르지만 결과는 거의 비슷하다고 한다. 또 No prop은 Least Mean Square Capacity(LMS Capacity)를 사용하는데 이것은 뉴럴 네트워크가 학습에서 Least Mean Square Error가 0이 될 수 있는 패턴의 갯수라고 한다. 아마도 학습 데이터에서 lms error가 0이 될 수 있는 뉴런의 갯수를 뜻하는 듯하다. No prop은 back prop을 단지 output layer에서만 한다. 즉 학습은 output layer에서만 하는 것이고 나머지 레이어에서는 weight를 무작위로 바꾸거나 고정시키며 학습을 한다. 이런 방식으로 학습이 된다니 신기하다.

하지만, 학습을 할 때 학습 패턴(뉴런의 갯수?)가 network capacity보다 작다면 no prop과 back prop은 거의 같은 성능을 보일테지만, 학습 패턴이 network capacity 보다 더 많다면, Back prop이 더 좋은 결과를 보인다고 한다. 또 이때는 network capacity를 늘림으로써 back prop의 성능을 뒤쫓을 수 있다고 한다.

논문의 저자는 No prop은 아직 연구 단계에 있는 알고리즘이이라 (당시 2013년에는) 어떤 상황에는 이 알고리즘을 쓰고 다른 상황에는 다른 알고리즘 쓰라고 하기에는 너무 이르다고 한다. 지금 no prop을 검색해봐도 그닥 여러 글은 나오지 않는다. 날로 발전하는 컴퓨터 파워에 결국 no prop의 필요성이 없어져 굳이 사용할 필요가 없었기에 수면 아래로 들어간 거 같다.

<a href="https://www.ncbi.nlm.nih.gov/pubmed/23140797">원문 보기</a>
