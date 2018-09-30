---
title: Hierarchical Question Image Co-Attention 리뷰
date: 2018-09-30
author: JuHyung Son
layout: post
tags:
  - paper
categories:
  - Deep_Learning
---

# Hierarchical Question Image Co-Attention for VQA
아카이브에 2017년에 올라온 HieCoAtt 모델입니다.
<center>

![](/image/HieVQA/8DE26CDA-124C-47F6-A1F1-98BF2579C607.png)

</center>
이번에는 Parikh 교수팀에서 나온 VQA 논문을 소개해 봅니다. 나온지는 좀 되었지만 처음으로 이미지와 텍스트 모두에 attention을 넣은 논문입니다.
이 논문의 가장 큰 기여는 두가지로 먼저 계층적 구조로 질문을 학습하는 것과 텍스트에도 attention을 넣었다는 것입니다. 질문을 계층적 구조로 만들었다는 건, 한 질문이 있으면, 단어별, 구문별, 문장 전체 이렇게 세가지로 구조화하여 학습을 하는 건데요. 이런겁니다.

<center>

![](/image/HieVQA/DEBD8402-F1FC-4C73-8614-B74736A042A8.png) {.aligncenter}

</center>
먼저 단어층에서 attention을 보고 그 다음은 구문층에서 봅니다. 마지막으로는 질문을 봅니다. 직관적으로 말하면, 단어층에서는 이미지에서 각 물체가 무엇이 있는지 보고, 구문층에서는 그것들 사이의 관계를 보고, 질문층에서는 이미지에 던진 질문을 보는 거죠.

## Method
다행히 저자는 논문에 쓰인 노테이션에 대해 친절히 설명해줍니다.
질문은 T개의 단어로 이뤄져 있다고 합시다. 각 t 번째 단어의 벡터를 $q_t$ 라고 하면 질문은 다음과 같이 나타낼 수 있죠.

$$Q = \{ q_1, q_2, ... ,q_T \}$$

라고 하죠. 세개의 계층적 구조를 만들것이니 위치 t 에서의 단어 임베딩, 구문 임베딩, 문장 임베딩을 다음처럼 표기합니다.

$$q^{w}_{t}, q^{p}_{t}, q^{s}_{t}$$

이미지 feature는 $V = \{ v_1 , ... v_N \}$ 이라고 하죠. 마지막으로 이미지와 질문에 대한 co-attention feature는 다음처럼 나타냅니다. $\hat{v}^{r}, \hat{q}^{r} , r \in \{ w, p, s \}$ 그리고 모든 웨이트들은 $W$ 로 표기합니다. 

이 노테이션을 가지고 질문의 계층적 구조 이해해봅시다.

먼저 one-hot 인코딩된 단어들, $Q = \{ q_1, ..., q_T \}$ 벡터 공간에 임베딩 시켜 단어 임베딩 $Q^w = \{ q^w_1 , ... ,q^w_T \}$ 를 얻을 수 있죠. 다음 구문 벡터를 얻어야죠. 그런데 구문은 다양한 길이의 단어들로 구성됩니다. 한개 일수도, 세개 일수도 있죠. 논문에서는 N-gram 방법을 사용합니다. 1, 2, 3 - gram 으로 1D convolution을 적용하는 것이죠. 그럼 각 윈도우 사이즈에서 나오는 구문 attention은 이렇습니다.

$$\hat{q}^{p}_{s, t} = tanh(W^s_c q^{w}_{t:t+s-1}), s \in \{ 1, 2, 3 \}$$

$W^s_c$ 는 문장에서의 웨이트이고 c는 뭔지 모르겠는데, c는 어디서 나온것이며 왜 단어에서의 웨이트가 아니고 문장에서의 웨이트인지는 모르겠네요. 여기서 각 n-gram을 1D conv에 넣을때 0-padded를 해주어 $\hat{q}^p_{1,t}, \hat{q}^p_{2,t}, \hat{q}^p_{3,t}$ 는 모두 같은 길이를 갖게 해줍니다. 이제 각 t 에서 max-pooling으로 구문 벡터를 얻습니다.

$$q^p_t = max(\hat{q}^p_{1,t}, \hat{q}^p_{2,t}, \hat{q}^p_{3,t}), t \in \{ 1, 2 ..., T \}$$

이렇게 구문 벡터까지 얻었습니다. 마지막으로 질문 벡터가 남았는데요. 질문 벡터는 구문 벡터를 LSTM으로 encoding 한 것의 hidden vector를 질문 벡터로 사용합니다. 그림으로 이해하면 이렇습니다.

<center>

![](/image/HieVQA/B109C00B-E0F4-4BA8-94F6-F2CEF278D92C.png)

</center>

이렇게 만든 계층적 구조로 나중에 대답을 내는 모델을 만들겁니다.
다음으로 Co-Attention을 봅시다. 이미지에만 attention을 적용한 VQA 연구에서 처음으로 질문에도 attention을 적용하였는데요. 저자는 2가지 co-attention 모델을 제시합니다. 첫째는 parallel co-att 로 이미지와 질문의 attention을 동시에 잡는 방법과 alternating co-att로 차례대로 attention을 잡는 방법이 있습니다. 논문 실험에서 좀 더 좋은 성능을 보이는 parallel 방법만 다뤄봅니다.

### Parallel Co-Attention
Parallel CoAtt 는 두가지 feature에 대한 attention을 동시에 잡습니다. 그러기 위해서 이미지와 질문 두가지를 합친 affinity matrix $C$ 를 만드는데요. 

이미지 feature map $V \in R^{d X N}$ , 질문 feature map $Q \in R^{d X T}$ 가 있다면, 다음처럼 affinity matrix를 만들어줍니다.

$$C = tanh(Q^{T} W_{b} V)$$

 이 때, 웨이트$W_b \in R^{ d X d}$ 이고 $C \in R^{T X N}$ 이 되겠죠. 즉 $C$ 의 row 는 질문 attention space를 이미지 attention space로 보내는 행렬이겠죠. 반대로 $C^T$ 는 이미지 attention space를 질문 attention space 로 보내줍니다.
이제 두 feature에서 attention을 동시에 뽑기위해 다음과 같은 레이어를 만들고 max값을 뽑아 attention 웨이트를 계산합니다.

$$H^{v} = tanh(W_{v} V + (W_{q} Q) C), H^{q} = tanh(W_{q} Q + (W_{v} V)C^{T})$$  

$$a^{v} = softmax (w^{T}_{hv} H^{v}), a^{q} = softmax(w^{T}_{hq} H^{q})$$

마지막으로 위의 attention 웨이트와 각 feature를 이용해 attention 벡터를 뽑아냅니다. 

$$\hat{v} = \sum^{N}_{n=1} a^v_n v_n , \hat{q} = \sum^{T}_{t=1} a^q_t q_t$$

그림으로 봐도 매우 헷갈리네요
<center>

![](/image/HieVQA/AA2CFFDE-BD8A-44E9-A1FC-1453EDA26624.png)

</center>
그러면 이렇게 얻은 것들로 질문에 답을 내는 모델을 봅시다. 

사실 모델은 상당히 단순합니다. 그냥 계층적 구조에서 MLP를 계속 쓰는 게 다죠. 단어수준에서 뽑은 attention을 fc 레이어에 넣어 단어 수준의 히든 레이어를 만들고 여기서 구문 수준의 attention을 concat 해 또 fc레이어에 넣고 마지막에는 문장 수준의 히든 레이어에서 softmax 로 답을 찾는 방법입니다. 모델 자체가 너무 단순해 성능이 생각보다 좋지 않은 건지 이 모델이 여러 모델 중 가장 좋은 성능을 보인건지는 모르겠지만 VQA 연구의 매우 초창기에 나온 연구기에 충분히 기여가 있는 연구입니다.
<center>

![](/image/HieVQA/5189BC97-D015-43C4-B8E1-477A1D82D434.png)

</center>