---
id: 290
title: 온라인 소스로 딥러닝 공부하기
date: 2017-11-10T17:15:35+00:00
author: JuHyung Son
layout: post
dsq_thread_id:
  - "6274562767"
image: /wp-content/uploads/2017/11/cropped-IMG_5212-1-250x250.jpg
categories:
  - Writing
---
<h2>온라인 소스를 이용하여 어떻게 혼자 박사 레벨의 기계학습, 딥러닝을 배울까?</h2>
Quora에 올라온 질문을 번역한 것입니다.

<a href="https://www.quora.com/How-can-I-learn-machine-learning-and-deep-learning-to-PhD-level-by-using-only-free-online-resources/answer/Abhishek-Patnia?share=351a369e&amp;srid=6jSd">Abhishek Patnia’s answer to How can I learn machine learning and deep learning to PhD level by using only free online resources? - Quora</a>
<h6>Abhishek Patnia, Applied Scientist at Amazon.com의 답변 내용</h6>
나의 조언은 스탠포드의 CNN 강의를 듣는 것부터 시작한다. 이 강의는 아래에 있는 모든 것들을 조금씩 담고 있고 당신이 아래의 조언을 조금씩 진행한다면 좋은 동기부여가 될 것이라고 생각한다.
<h5><strong>기본 지식</strong></h5>
<h6><em><strong>수학</strong></em></h6>
선형대수, 확률론, 미적분을 공부해야 한다. 모든 이론은 공부하기보단 이것들을 도구의 관점에서 바라보고 접근하자. 어떤 문제에 마주쳤을 때, 그것에 관련된 주제를 배우고 왜 그 주제가 중요한지 명확히 하자.

- <a href="http://www.deeplearningbook.org/contents/part_basics.html">Deep learning book의 첫번째 장</a>. 수학적인 개념과 관련하여 기계학습을 아주 좋은 설명으로 설명해 놓았다.
- <a href="https://www.youtube.com/watch?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&amp;v=kjBOesZCoqc">Essence of Linear Algebra</a> &amp; <a href="https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr">Essence of Calculus</a>. 몇 시리즈로 구성된 유투브 비디오로 선형대수와 미적분을 직관적으로 쉽게 설명한다.
- <a href="http://djm.cc/library/Calculus_Made_Easy_Thompson.pdf">Calculus Made Easy by Thompson</a>. 난 최근 이 책을 보기 시작했고 아주 만족스럽다. 미적분에 대한 아주 좋은 시작서이다.
- <a href="https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/video-lectures/">MIT Probability Systems Analysis &amp; Applied Probability</a>. 나에겐 조금 느리긴 했지만 모든 주제를 다룬다.
- <a href="https://graphics.stanford.edu/courses/cs205a-13-fall/assets/notes/cs205a_notes.pdf">Mathematical Methods for Computer Vision, Robotics, and Graphics Course Notes</a>. Khan Academy에서 보충자료를 받을 것을 추천한다.
- <a href="https://ocw.mit.edu/courses/mathematics/18-085-computational-science-and-engineering-i-fall-2008/index.htm">Mathematical Methods for Engineers 1 and 2. MIT</a>의 응용수학 강의다. 몇몇은 AI에 관련되있고 아닌 것도 있다. 나는 몇몇 강의를 선택하여 보았다.
<h6>
<em><strong>소프트웨어 엔지니어링</strong></em></h6>
- 파이썬. 대부분의 AI 라이브러리와 샘플 코드는 파이썬으로 작성되어 있다.
- 파이썬에 익숙해졌다면 병렬 프로그래밍에 친숙해지자. 학습 데이터를 만들때나 계산을 빠르게 할 때에 필수로 사용하여야 한다. 이 <a href="http://sebastianraschka.com/Articles/2014_multiprocessing.html">블로그</a>가 시작하기 좋다. GIL이라는 멋진 세계를 만날 것이다.
- <a href="http://www.numpy.org/">Numpy</a>와 <a href="https://www.scipy.org/">Scipy</a>의 고수가 되자. 이 두가지 도구는 수학에 관련된 것들이다. 스탠포드의 이<a href="http://cs231n.github.io/python-numpy-tutorial/"> 자료</a>로 시작하면 좋다. 간단한 설명을 하면, Numpy는 컴퓨터에서 선형대수를 다루는 것이다. 대부분의 AI 도구를 다루기 편해질 것이다.
<h5>
<strong>GPU 가속</strong></h5>
대부분의 기계학습은 행렬로 표현되어 있다. 행렬은 GPU를 통해 엄청나게 가속 된다. ML 모델 가속을 위해 GPU를 보는 것을 추천한다.

- 장기적인 관점에서 너만의 GPU 시스템을 만드는 것이 더 저렴하다. 이 가이드들을 참조하자.<a href="http://timdettmers.com/2015/03/09/deep-learning-hardware-guide/"> A Full Hardware Guide to Deep Learning - Tim Dettmers</a>, <a href="https://medium.com/towards-data-science/building-your-own-deep-learning-box-47b918aea1eb">Building your own deep learning box – Towards Data Science – Medium</a>.
- 이 <a href="https://www.amazon.com/CUDA-Example-Introduction-General-Purpose-Programming/dp/0131387685/ref=sr_1_1?ie=UTF8&amp;keywords=cuda+by+example&amp;qid=1494789351&amp;s=books&amp;sr=1-1">책</a>과 이 <a href="https://www.udacity.com/course/intro-to-parallel-programming--cs344">강의</a>를 추천한다. 하지만 이 모든 것을 알 필요는 없고 단지 친숙해지기만 하면, 너의 네트워크를 빠르게 돌릴 수 있고 디버깅도 잘하게 될 것이다.
- 적어도 하나의 GPU based framework를 배우자. <a href="http://mxnet.io/">MXNet</a>, <a href="http://deeplearning.net/software/theano/">Theano</a>, <a href="https://www.tensorflow.org/">Tensorflow</a>와 같은 것들이 많이 있다. 이것들은 기본적으로 선형대수 엔진이다. 웹사이트에 가이드와 튜토리얼이 많다.
<h5><strong>기계학습</strong></h5>
기계학습은 꽤 큰 분야이다. 이 분야는 당신이 어떤 것을 추구하냐에 따라서 기계학습의 종류가 달라질 것이다.나는 인공신경망을 선호하므로 내 조언이 꽤 치우쳐 있을 수 있음을 감안하자.
<h6><em><strong>강의</strong></em></h6>
- <a href="https://www.youtube.com/watch?index=1&amp;list=PLCA2C1469EA777F9A&amp;v=VeKeFIepJBU">Learning from Data</a>
- <a href="https://www.coursera.org/learn/neural-networks">Neural Networks for ML</a>
- <a href="http://cs231n.stanford.edu/">CNN for visual recognition</a>
-<a href="http://cs224d.stanford.edu/"> DL for NLP</a>
- <a href="https://github.com/oxford-cs-deepnlp-2017/lectures">Oxford CS Deep NLP</a>
- <a href="https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH">Neural Networks</a>
<h6>
<em><strong>블로그</strong></em></h6>
- <a href="http://www.deeplearningbook.org/">Deep Learning Book</a>
- <a href="http://karpathy.github.io/">Andrej karapathy</a>
- <a href="http://colah.github.io/">Colah</a>
- <a href="http://sebastianruder.com/">Sebastian Ruder</a>
- <a href="http://www.wildml.com/">WildML</a>
- <a href="http://distill.pub/">Distill</a>
<h6><em><strong>논문</strong></em></h6>
- <a href="https://github.com/terryum/awesome-deep-learning-papers">Great list compiled by Terry Taewoong Um</a>
- <a href="http://www.arxiv-sanity.com/">Create an account on Arxiv Sanity Preserver and well go insane</a>

이 분야는 매우 빠르게 발전한다. 늦춰지지 않기 위한 가장 좋은 방법은 이 분야의 리더들을 팔로하는 것이다. 그리고 그 가장 좋은 팔로 방법은 소셜 미디어를 사용하는 것이다. 다음 사람들을 팔로해보자.
- Andrew Ng
- Yoshua Bengio
- Yann LeCun
- Fei-Fei Li
- Andrej karapathy
<h6>
<em><strong>자신만의 자료들을 만들자.</strong></em></h6>
나는 내 자료들을 어떻게 모을 지 계속 고민한다. 한번은 논문들을 아무데나 프린트해놨다. 그리고나선 아이패드 프로로 paperless를 시도했지만 다시 종이를 사용했다. 결국 몇달 전 나만의 방법을 찾은 것 같다.
- PDF를 관리할 때 Papers를 사용한다. Arxiv 같은 곳에서 PDF를 불러올 수 있다.
- Google Keep으로 북마크를 해둔다. 그리고 나서 시간이 나면 본다.
