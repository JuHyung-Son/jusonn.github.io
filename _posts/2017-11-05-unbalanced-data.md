---
id: 195
title: 불균형 데이터에 대한 7가지 팁
date: 2017-11-05T21:58:24+00:00
author: JuHyung Son
layout: post
permalink: '/%eb%b6%88%ea%b7%a0%ed%98%95-%eb%8d%b0%ec%9d%b4%ed%84%b0%ec%97%90-%eb%8c%80%ed%95%9c-7%ea%b0%80%ec%a7%80-%ed%8c%81/'
dsq_thread_id:
  - "6264395222"
image: /wp-content/uploads/2017/11/bulb-664657.jpg
categories:
  - Writing
---
<h1>7 Techniques to Handle Imbalanced Data</h1>
이 포스트는 매우 불균형한 데이터를 다루는 방법에 대한 것입니다.
kdnuggets.com의 포스트를 번역한 것입니다. <a href="http://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html?utm_content=bufferf0775&amp;utm_medium=social&amp;utm_source=facebook.com&amp;utm_campaign=buffer">원문보기</a>

By Ye Wu &amp; Rick Radewagon, IE business school
<h2>Introduction</h2>
은행 사기 예측, 실시간 경매나 네트워크에서의 공격 감지와 같은 분야에서의 데이터들은 주로 어떤 특징을 가졌는가?

이런 분야에서의 데이터는 1%도 안되는 '흥미로운' 사건을 가진다.(e.g. 신용카드 사기, 자신의 광고 클릭) 그러나 대부분의 기계학습 알고리즘은 이런 불균형 데이터에서 그 힘을 잘 발휘하지 못한다. 이 7가지 방법은 불균형 데이터에서의 분류를 학습하는데 도움이 될 것이다.  <img class="aligncenter wp-image-199 size-full" src="http://dllab.xyz/wp-content/uploads/2017/11/imbalanced-data-1.png" alt="" width="652" height="180" />
<h2>1. 상황에 맞는 평가 방법을 사용하라</h2>
불균형한 데이터를 사용한 모델에 적합하지 않은 평가 방법을 사용하는 것은 위험하다. 위 그래프와 같은 경우가 있다고 생각해보자. 이 모델에서 정확도를 측정한다면 모든 샘플에 대해서 '0'을 매기는 모델은 99.8%의 정확도를 가질 것이다. 물론 이 모델은 우리에게 아무 정보도 주지 못하는 모델이다.
이런 경우엔 다른 평가 방법을 적용해야 한다.

&nbsp;
<ul>
 	<li>Precision/Specificity : 선택된 인스턴스 중 몇 개가 관련있는지.</li>
 	<li>Recall/Sensitivity : 관련된 인스턴스 몇 개가 선택되었는지</li>
 	<li>F1 score : precision과 recall의 조화평균</li>
 	<li>MCC : 실제값과 예측값의 상관계수</li>
 	<li>AUC : TP의 비율과 FP의 비율의 관계</li>
</ul>
&nbsp;
<h2>2. 데이터 리샘플링</h2>
적합한 평가 방법을 선택하는 것과 다른 데이터셋을 얻는 방법도 있다. 불균형한 데이터로 부터 균형된 데이터를 얻는 방법은 under-sampling과 over-sampling이 있다.
<h2>2.1 Under-sampling</h2>
Under-sampling은 상대적으로 많은 클래스의 크기를 줄임으로 데이터를 균형 맞춘다. 이 방법은 데이터가 충분히 클 때 사용한다. 상대적으로 적은 클래스의 데이터와 같은 크기의 많은 클래스의 데이터를 뽑는다. 이 과정에서 같은 크기의 데이터를 뽑는 과정에는 여러가지가 있다.
<h2>2.2 Over-sampling</h2>
반대로, over-sampling은 데이터가 크지 않을 때 사용하는 방법이다. 상대적으로 적은 클래스의 데이터를 늘림으로 데이터를 균형되게 한다. 데이터를 늘리는 데는 repetition, bootstrapping, SMOTE와 같은 방법이 쓰인다.
<h3>3.K-fold CV 사용</h3>
Over-sampling을 할 때에 cross-validation이 적용된다는 것도 주목할만하다.
over-sampling이 실제값과 bootstrapping으로 분포 함수에 의해 새 데이터를 뽑는다는 것을 기억해야한다. over-sampling을 한 후 cross-validation을 하는 것은 모델을 overfitting하는 것이다. 그래서 cross-validation은 over-sampling하기 전에 사용되어야한다. feature selection을 하는 것처럼 말이다. 데이터를 반복적으로 sampling함으로 데이터는 overfitting 문제가 없는 무작위성(Randomness)을 가진다.
<h3>4. 서로 다른 sampled 데이터를 앙상블하라</h3>
좋은 모델을 만드는 가장 쉬운 방법은 데이터를 더 쓰는 것이다. 문제는 logistic regression이나 random forest와 같은 out-of-the-box 분류기는 상대적으로 적은 클래스를 버린다는 것이다. 가장 좋은 방법은 적은 클래스의 데이터와 많은 클래스의 서로 다른 n개의 데이터 샘플을 모두 사용하여 n개의 모델을 만드는 것이다. 10가지의 모델의 ensemble 모델을 만든다고 생각해보자.
1000개의 적은 클래스 샘플과 랜덤한 10,000개 많은 클래스 샘플이 있다면, 10,000개의 샘플을 10개로 나눌 것이다.

이 방법은 원본 데이터가 크다면 간단하고 수평적으로 확장 가능한 방법이다.

<img class="aligncenter wp-image-198 size-full" src="http://dllab.xyz/wp-content/uploads/2017/11/R1920x0-13.png" alt="" width="728" height="355" />
<h3>5. 다른 비율로 resample하기</h3>
4. 의 방법은 적은 클래스와 많은 클래스의 비율을 이용해 미세조정 할 수 있고 최적의 비율은 데이터와 모델에 달려있다. 하지만 앙상블에서 같은 비율로 모든 모델을 학습하는 것보단 다른 비율로 앙상블을 학습하는 것이 나을 수 있다. 10개의 모델이 학습된다면, 1:1(적은:많은 클래스)의 비율이 적당하고 다른 모델은 1:3이나 2:1이 될 수 있다. 학습에 쓰인 모델에 따라서 이 방법은 하나의 클래스의 가중치에 영향을 줄 수 있다.

<img class="aligncenter wp-image-197 size-full" src="http://dllab.xyz/wp-content/uploads/2017/11/imbalanced-data-3.png" alt="" width="346" height="464" />
<h3>7. 자신만의 모델을 만들어라</h3>
이전의 6가지 방법은 데이터 자체에 초점을 두었고 모델을 고정된 것으로 보았다. 하지만 실제로 모델이 불균형한 데이터에 맞다면 데이터를 resample 할 필요는 없다. 그 유명한 XGBoost는 클래스가 너무 치우져있지 않다면 사용하기에 좋은 출발점이다. XGBoost 자체가 데이터가 불균형되지 않게 하기 때문이다.
모델이 분류를 할 때에 많은 클래스의 에러보다 적은 클래스의 에러에 더 가중치를 주게 cost function을 만든다면, 모델이 알아서 적은 클래스를 다루게 할 수 있다. 예를 들어 SVM을 조작하여 오분류에 대한 가중치를 더 줄 수 있다.

<img class="aligncenter wp-image-196 size-full" src="http://dllab.xyz/wp-content/uploads/2017/11/imbalanced-data-4.png" alt="" width="353" height="200" />
<h3>끝으로</h3>
이 6가지가 모든 것이라기보단 불균형 데이터를 다루는 시작점이라고 할 수 있다. 모든 문제에 최적인 방법이나 모델은 존재하지 않기 때문에 모든 방법과 모델을 테스트하여 어떤 것이 가장 잘 작동하는지 확인해야한다. 창의적으로 여러 방법을 써보아라. 데이터의 불균형이 많이 일어나는 domain에 대한 지식을 갖고 있는 것 역시 중요하다. "market-rules"는 항상 변한다. 그러니 이미 사용하던 방법이 계속 쓰이는지 항상 확인해보아라.

<b>Ye Wu</b> is pursuing the Master in Business Analytics &amp; Big Data at the IE Business School. She has a background in Accounting and hands-on experience in Marketing and Sales Forecasting.
<b>Rick Radewagen</b> is an aspiring Data Scientist with a background in Computer Science. He is also pursuing the Master in Business Analytics &amp; Big Data at the IE Business School.
<h3>Related:</h3>
&nbsp;
<ul>
 	<li><a href="http://www.kdnuggets.com/2016/08/learning-from-imbalanced-classes.html">Learning from Imbalanced Classes</a></li>
 	<li><a href="http://www.kdnuggets.com/2016/04/unbalanced-classes-svm-random-forests-python.html">Dealing with Unbalanced Classes, SVMs, Random Forests, and Decision Trees in Python</a></li>
 	<li><a href="http://www.kdnuggets.com/2017/01/tidying-data-python.html">Tidying Data in Python</a></li>
</ul>
&nbsp;
