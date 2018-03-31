---
id: 168
title: 'Hybrid decision tree, Naive Bayes Classifier &#8211; Paper review'
date: 2017-11-05T21:18:44+00:00
author: JuHyung Son
layout: post
permalink: /hybrid-decision-tree-naive-bayes-classifier-paper-review/
dsq_thread_id:
  - "6264392248"
image: /wp-content/uploads/2017/11/R1920x0-2-250x250.png
categories:
  - Paper
---
<h2>Hybrid decision tree and naive Bayes classifier for multi-class classification tasks, Dewan Md. Farid et tal.</h2>
이 논문은 Decision tree와 Naive Bayes classifier 의 정확성을 향상시키는 방법에 대한 논문이다. 논문 제목에서 볼 수 있듯이 저자는 이 둘은 결합한 알고리즘을 제안한다.

Decision tree(DT)와 Naive Bayes Classifier(NB)는 둘다 오랫동안 자주 쓰인 알고리즘이다. 오랜 기간 쓰인 만큼 알고리즘도 여러가지 변형된 모델이 나오기도 했지만 여기서는 가장 기본적인 DT와 NB를 다룬다.

이 두 알고리즘은 기본적으로 Supervised Classification이다. 라벨이 있는 데이터로 학습을 하고 라벨이 없는 데이터를 분류해 라벨을 매기는 일을 하는 것이다. 흔히 쓰이는 DT는 ID3로 정보 이론을 사용하여 분류를 한다. tree를 나누면서 정보량을 계산하여 그 이전의 정보량과 차이를 계산한다. 정보량을 계산하는 것에도 역시 여러가지 방법이 있으나 다루지는 않겠다. 이것을 반복하여 tree를 나누는게 정보의 관점에서 아무런 의미가 없을 때, 과정을 멈춘다. NB는 말 그대로 Bayes theorem을 기반으로 한 classifier이다.

<a class="tx-link" href="https://en.wikipedia.org/wiki/Bayes%27_theorem" target="_blank" rel="noopener">Bayes theorem은 이곳</a>

논문은 두가지 hybrid algorithm을 제안한다. 먼저는 hybrid decision tree algorithm이다. 이 알고리즘은 크게 말해 NB를 이용해 outlier를 제거하고 decision tree를 적용하는 방법이다. 훈련 데이터에 먼저 NB로 학습을 시킨 후 각 클래스 별로 따로 학습한 NB를 적용한다. 다음 각 클래스에서 잘못 분류된 데이터들이 생기면 그것을 제거하고 DT를 사용하는 것이다.<img class="aligncenter wp-image-172 size-full" src="http://dllab.xyz/wp-content/uploads/2017/11/R1920x0-2.png" alt="" width="602" height="809" />

또 다른 알고리즘인 hybrid naive bayes algorithm도 별반 다르지 않다. 이번엔 훈련 데이터에 먼저 DT를 학습한다. 학습된 DT를 보면 학습에 쓰인 feature와 쓰이지 않은 feature들이 있을 것이다. 여기서 중요한 feature들만을 사용해 NB를 적용시킨다. 간단하게 feature selection을 DT를 이용해 하는 것이다.

<img class="aligncenter wp-image-173 size-full" src="http://dllab.xyz/wp-content/uploads/2017/11/R1920x0-6.png" alt="" width="404" height="262" /><img class="aligncenter wp-image-171 size-full" src="http://dllab.xyz/wp-content/uploads/2017/11/R1920x0-3.png" alt="" width="402" height="389" />

처음 이 논문을 보고 hybrid naive bayes는 중요한 feature를 고르는 것이니 합리적이다라고 생각했지만 outlier? 라고 여겨지는 것들을 제거하는 hybrid DT는 합리적인지 모르겠다고 생각했다. NB로 걸러지는 것들이 outlier라고 하기엔 좀 그렇고 NB가 완벽한 classifier도 아닌데 그냥 제거해도 되는 것인가 하는 생각이 든다.

하지만 논문에서 실행한 실험들을 보면 이 hybrid 기법들이 월등히 좋은 정확성을 보여준다. 단순히 accuracy 뿐만아니라 recall, precision에서도 더 좋은 값을 가진다. 데이터데 알고리즘에 특화된 데이터라 결과가 좋다고 생각된다. 실제로 이 방법을 사용할 때엔 좀 더 조심할 필요가 있을 듯 하다.

<img class="aligncenter wp-image-170 size-full" src="http://dllab.xyz/wp-content/uploads/2017/11/R1920x0-4.png" alt="" width="603" height="350" /><img class="aligncenter wp-image-169 size-full" src="http://dllab.xyz/wp-content/uploads/2017/11/R1920x0-5.png" alt="" width="614" height="792" />
