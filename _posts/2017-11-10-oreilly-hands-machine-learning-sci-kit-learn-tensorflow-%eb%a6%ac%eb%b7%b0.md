---
id: 283
title: 'O&#8217;REILLY Hands on machine learning with Sci-kit learn and tensorflow 리뷰'
date: 2017-11-10T01:07:10+00:00
author: JuHyung Son
layout: post
permalink: '/oreilly-hands-machine-learning-sci-kit-learn-tensorflow-%eb%a6%ac%eb%b7%b0/'
dsq_thread_id:
  - "6273078875"
image: /wp-content/uploads/2017/11/cropped-nature-2609118-1-250x250.jpg
categories:
  - Daily life
---

<img class="aligncenter wp-image-284 size-full" src="http://dllab.xyz/wp-content/uploads/2017/11/11.jpg" alt="" width="727" height="783" />

3개월 정도 이 책을 보는 중이다. 진도는 천천히 나가면서 필요할 때 마다 찾아서 읽어보고 코드를 찾아보는 정도로 읽는 중이다. 이제 챕터 11까지 끝내고 CNN과 RNN을 남겨두고 있고 여기까지의 리뷰를 짤막하게 작성한다.
<h4>1. The machine learning Landscape</h4>
다른 책들과 마찬가지로 머신러닝이 무엇인지 개념부터 잡는다. 그리고 머신러닝이 다루는 문제와 머신러닝에 대한 전체적인 틀을 보는 챕터이다. 이전에 머신러닝에 대해 공부를 해보았다면 넘어가거나 가볍게 읽는 정도로 넘어갈 수 있다.
<h5>2. End-to-End Machine learning project</h5>
LA 집값 데이터를 가지고 실제 프로젝트를 해본다. 데이터 전처리부터 Fine Tuning 까지 정말 처음부터 끝까지 보여준다. 데이터를 전처리하는 부분에 신경을 많이 쓴 듯하다. 한 과정에도 여러가지 방법을 보여주는데 이 부분이 도움이 된다. 코드 자체가 매우 고급지게 짜였다. 이런 방식으로 코드를 짜는 것을 연습해보고 있다. 그렇지만 데이터가 워낙 단순해 실제로 비슷한 코드를 짜는데는 연습이 필요할 거 같다. 가장 좋은 점이 간단한 과정에도 고급진 코딩을 볼 수 있다는 것이다. 마치 캐글 kernel을 보는 듯 하다.
<h4>3. Classification</h4>
유명한 MNIST 데이터를 가지고 분류를 해본다. Classification의 기초는 performance measure이다. 역시나 이 장은 measuring에 신경을 썼다. 이 measuring 부분은 중요한 부분이니 자세히 보는 것이 좋다. 알고리즘 자체는 SVM이나 K Neighbor과 같은 코드를 돌리기만 하면 되는 것이라 간단하게 소개하고 차이를 보여준다. 제목은 classification이지만, classification performance measure로 생각하면 된다.
<h4>4. Training Models</h4>
모델링 방법들에 대해 소개하는 장이다. 처음엔 2장이 regression에 관한 것인줄 알았는데 2장은 머신러닝 과정에 대해 소개만 하는 것이고 이 4장이 regression에 대한 설명이다.  GD, Regularized linear models와 같은 가장 기본적인 regression을 다룬다. sci-kit learn을 이용하여 코드에 대한건 그닥 자세히 나오지는 않는다. 그보단 알고리즘에 대한 소개, 즉 GD, SGD, Ridge이 무엇인지에 더 신경을 썼다. 이 부분은 코드가 별로 없고 설명 뿐이라 대충 넘어갈 수도 있으나 사실 매우 중요한 부분이다. Regression에 대한 기본 개념을 잡자. 학습이라는 것이 무엇인지, overfitting, underfitting에 대한 깨달음? 을 얻자. 또 딥러닝을 하기 위해선 이 부분을 매우 잘 이해하고 넘어가야한다.
<h4>5,6. SVM, Decision Tree</h4>
SVM이 어떻게 작동되는지, 왜 좋은 알고리즘인지 소개한다. 자세한 설명보다는 큰 맥락으로 이러하게 돌아간다 는 것을 보여주는 거 같다. 사실 수학적으로 깊게 들어갈 것이 아니면 어떻게 작동하는지만 알면 되는 알고리즘이라 많은 설명이 있진 않다.
<h4>7. Ensemble Learning and Random Forests</h4>
머신러닝 프로젝트의 끝에는 앙상블을 쓰기 마련이다. 여기서 그 앙상블을 다루는데 설명이 아주 좋다. 사실 머신러닝에 대한 큰 개념이 없으면 처음부터 앙상블을 적용하거나 그냥 이유없이 쓰는 경우가 많다. 앙상블을 왜 쓰는지, 어떻게 적용하는지 설명이 되어있다. 단순히 앙상블 뿐만 아니라 RF, Ada, GB 도 어떻게 regression을 하고 classification을 하는지 보여준다.
<h4>8. Dimensionality Reduction</h4>
차원 축소는 내가 아직 딱히 쓸 데가 없어 개념만 아는 정도이다. 여기선 PCA를 주로 다루고 나머지 기법은 짧게 소개만 한다. 그닥 자세한 내용인지는 잘 모르겠다.
<h4>9. Up and Running with TensorFlow</h4>
여기서부터 딥러닝을 시작한다. 9장은 딥러닝보다는 텐서플로우를 설명하는 장이다. 텐서플로를 사용한 linear regression부터 GD, tensorboard등 주로 텐서플로 사용법인데 몇몇은 아직도 어렵다. 이 장은 코드를 따라해보며 텐서플로가 뭔지, 어떻게 사용하는지 간단하게 알고 넘어가보았다.
<h4>10. Introduction to Artificital Neural Network</h4>
이 부분부터가 이 책의 꽃이라고 나는 생각한다. 여기서는 DNN을 짜는 과정인데 초반에 나오는 High-Level API는 별로 안쓰이므로 넘어갔다. 다음으로 텐서플로를 사용해 DNN코드는 짜는데 코드가 예술이다. DNN 코드 짜는 것 자체가 예술성이 있다고 나는 생각하지만, 이 책보다 좋은 DNN코드는 지금까지 못봤다. (tensorflow tutorial, Sung Kim님의 모두의 머신러닝등 여러가지를 보았다.) DNN코드가 정말 깔끔하고 잘 짜여져 있어 공부하기에 매우 좋다.
<h4>11. Training Deep Neural Nets</h4>
10장에서 DNN의 초기 모델을 짠다면 여기는 모델을 튜닝하는 장이다. 딥러닝에는 activation function만 해도 여러가지이고 optimizer, regularizer에도 종류가 많아 큰 바다에 빠지는 느낌인데 모델을 짜는데 있어 좋은 방향을 준다. 코드를 짤때 참고하기도로 매우 좋다.

이 책의 강점은 코드라고 생각한다. 내가 프로그래밍을 못해서 인지 코드를 보며 감탄하기도 한다. 그리고 머신러닝을 처음 접하는 독자에겐 약간 어려울 수 있는 난이도라고 생각한다. 시중에 있는 많은 책들이 초보자용인 경우가 많아 그 부분을 걱정했는데 그 정도는 아니다. 다음엔 CNN, RNN, Autoender 파트가 있다. 이 부분은 다른 곳에서 좀 더 개념을 잡은 후에 보려고 계획 중이다.
