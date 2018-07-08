---
title: The Amazing Mysterious of the gutter 리뷰
date: 2018-07-08
author: JuHyung Son
layout: post
categories:
  - Paper
---

# The Amazing Mysteries of the Gutter: Drawing Inferences Between Panels in Comic Book Narratives

> 놀라운 **Gutter**(물이 흐른 자국, 도랑) 의 미스테리  

이번에 볼 논문은 미국 만화 데이터를 이용한 만화 QA 논문입니다. 이 논문은 만화라는 도메인을 쓴 몇 안되는 논문이고 또 직접 데이터셋을 만들어 배포하고 있습니다. 이후 만화 도메인에 관한 연구가 나온다면 MNIST 와 같은 연구 데이터로도 쓰이지 않을까 하네요.

# Abstract
만화라는 장르는 우리에게 전달되는 정보(그림, 텍스트)와 만화가에 의해 생략된 것들의 조합으로 이루어져 있습니다. 그리고 만화가 재밌는 이유는 이 생략된 부분을 독자가 직접 채우면서 읽어야하기 때문이죠.

만화의 그림과 텍스트를 읽으며 만화를 이해하는 것을 **closure** 라고 합니다. 이 closure 를 통해 생략된 부분(gutter)를 계속 추론하는 것이죠. 그리고 이 과정은 상당히 non-trivial 합니다. 일반적인 이미지와 텍스트는 이미 연구가 많이 되어 있지만, 만화의 제각각 스타일과 연속적이지 않은 텍스트는 아직 연구가 거의 되지 않았습니다. 그래서 저자는 만화라는 closure-driven narrative 를 모델이 이해할 수 있게 만드는 것이 목표라고 합니다.

그리고 **만화를 이해했다** 라는 건 빈칸 채우기 문제를 통해서 이해의 정도를 측정합니다. 수능에서 언어 문제를 풀 듯이요.

# Intro

![](/image/gutters/1.png)

**closure** 를 제대로 보면, (1) 각각의 패널(컷)을 이해하고 (2) 패널들 사이의 연결고리를 추론하는 과정입니다. 그냥 우리가 만화책을 보는 과정으로 이해하면 됩니다.

저자는 컴퓨터에게 closure 는 매우 챌린징한 일이라고 합니다. 왜냐면 만화의 물체들은 만화가에 의해 추상화되어 있기도하고 stylized 되어 있기도 하기 때문이죠. 또 만화에는 텍스트를 읽는 순서가 명시되어 있지 않습니다. 이런 것들이 만화를 이해시키는데에 가장 큰 어려움이라고 합니다.

![](/image/gutters/2.png)

# Dataset
이 논문의 가장 큰 기여는 데이터셋을 만든거라고 생각하는데요. 저자는 4천권의 만화책으로부터 1. 컷들을 추출하고 2. 말풍선의 위치와 텍스트를 추출하고 순서를 매겨놓았습니다. 또 QA를 위한 문제와 컷들에 관련된 메타데이터도 만들었죠.

![](/image/gutters/6.png)

# Data analysis
> from 만화의 이해 - Scott McCloud  

컴퓨터에게 만화를 이해시키려면, 일단 만화가 뭔지 알아야겠죠. 저자는 _Scott McCloud - 만화의 이해_ 라는 책에서 만화에 대한 정보를 줍니다. 한국어로도 있는 책이네요.

만화를 이해하는데는 크게 이 두가지가 있습니다.

**intrapanel** : 하나의 패널안에서 이미지와 텍스트의 상호작용

**interpanel** : 한 패널 안에 내래티브(이미지, 텍스트)가 어떻게 다음 패널로 연결되는지

그리고 이 두가지 종류는 다음의 카테고리로 나뉩니다.

## IntraPanel
1. Word-Specific(4.4%): 이미지가 있긴 하지만 텍스트에 더 대부분의 정보가 있음
2. Picture-specific(2.8%) : 이미지가 대부분의 정보를 줌
3. parallel(0.6%) : 텍스트와 이미지가 interact 하지 않고 따로 놈.
4. Interdependent(92.1%) : 텍스트와 이미지가 같이 정보를 전달
## InterPanel
1.  moment to moment 0.4% : 영화의 1 프레임 움직이는 것처럼 단순 동작이 움직임
2. action to action 34.6% :  액션을 통해 내용이 진행됨
3. subject to subject 32.7% : 같은 씬에서 내용이 진행되는 중 새로운 subject 발생
4. scene to scene 13.8% : 시간, 공간에서 큰 변화가 일어남.
5. continued conversation 17.7% : 그림 그대론데 말만 바뀜.

![](/image/gutters/5.png)

# Tasks : Test the performance of closure
논문의 실험은 3가지로 진행됩니다. 단순한 분류라기 보다는 만화라는 데이터를 얼마나 이해했느냐를 보는 과정입니다.

1. 텍스트 빈칸 채우기
2. 이미지 빈칸 채우기
3. character coherence (대화 말풍선 순서 맞추기)

결국 모두 다 패널 $p_{i-1}, p_{i-2} ,.... , p_{i-n}$ 가 주어졌을 때 $p_i$ 를 예측하는 task 로 볼 수 있습니다.
모델이 만화를 이해한 것인지를 보기 위해서는 이런 문제를 푸는 것이 평가하기에 적합하다고 합니다.

다음 모델을 만들기에 앞서 이슈가 하나 있습니다. 만화 컷들을 그대로 학습하다보면 컷들에 있는 텍스트를 학습하게 되어 아주 약한 OCR 이 학습될 수 있다는 겁니다. 그렇게 되면 문제를 푸는데 좀 치트키가 될 수 있죠. 그래서 학습시에 모든 텍스트 박스는 검은색으로 처리했다고 합니다.

![](/image/gutters/3.png)
# Model
이 논문에선 Hierarchical LSTM 구조가 제일 좋은 성능을 보여줍니다.
![](/image/gutters/7.png)
(두번째 패널 오타)

패널 $p_i$ 패널의 이미지 $z_i$ 텍스트 $t_{i_{x}}$ 바운딩 박스 $b_{i_{x}}$

**image-text model**
- $t_i$ 각 텍스트 박스의 word embedding sum 후 intra panel LSTM.
- $z_i$ VGG 16 feature 사용
- t, z 를 fc 를 거친 후 interpanel LSTM에 넣음

![](/image/gutters/4.png)

보시면 성능이 그리 좋지는 않죠. 아직 연구할 것이 많은 분야인 거 같습니다.
