---
title: Gensim 이용한 Word Embedding, Word2vec
date: 2018-04-24
author: JuHyung Son
layout: post
tags:
  - NLP
  - Word embedding
  - gensim
categories:
  - Deep_Learning
---

텐서플로우는 Embedding을 이용해 임베딩을 한 후 신경망 모델에 넣는 전처리용으로는 적합합니다. 그런데 단어 자체를 분석하고 토픽 모델링을 하기엔 제공되는 모듈이 많이 없어 어느정도 제한적이죠. Gensim 은 이런 작업에 아주 편리한 파이썬 라이브러리입니다. Gensim의 목적은 Topic modelling for humans 자연어 처리에 특화되있습니다. 설치할 때 씨 컴파일러가 있어야 씨로 짜여진, 약 70배 빠른 버전을 설치할 수 있다네요.

<a href="https://radimrehurek.com/gensim/index.html">Gensim </a>

저번에 했던 word2vec을 이어 gensim을 이용한 word2vec을 해보았습니다. 데이터는 현재 진행 중인 프로젝트 데이터를 사용하였고 트위터 텍스트입니다. 트위터에서 나타나는 루머들이 데이터상으로 어떤 형태를 띄나 보려고합니다.

<div align="center"> <img src="/image/gensim/1.png" /> </div>

제가 임베딩할 부분은 사용자들이 날린 트윗 내용으로 총 2813809개의 단어로 이루어져있습니다.

<div align="center"> <img src="/image/gensim/5.png" /> </div>

gensim은 텍스트를 gensim의 인풋 형태로 변형해주며 그외 자잘한 처리들까지 해주는 모듈도 있습니다.
`gensim.utils.simple_preprocess` 그래서 데이터를 처리할 때는 트위터 텍스트를 위 데이터프레임과 같이 처리하기 쉽게하는 과정만 거치면 됩니다.
위와 같이 말뭉치까지 처리가 완료되면 임베딩을 하는데에는 한줄이면 됩니다.

`model = gensim.models.Word2vec(sentences, size=300, min_count=7)`

300차원의 임베딩 행렬을 만들었습니다. `min_count=7` 은 7번 이상 나온 단어들에 대해서만 학습을 한다는 겁니다.
학습이 구글의 word2vec <a href="https://code.google.com/archive/p/word2vec/">C 패키지 </a> 를 사용해서인지 epoch이 작은건지 매우 빠릅니다.

 <div align="center"> <img src="/image/gensim/2.png" /> </div>
 사과의 벡터를 뽑아봅니다. 이렇게 쉽게 임베딩 행렬을 다룰 수 있습니다.

 또 논문에서처럼 쉽게 단어 벡터를 가지고 연산을 할 수 있습니다. 사과 + 아이폰 해보면 이러한 결과가 나오기도 합니다. 꽤 수긍이 가지만 다른 단어들의 연산에 대해서는 엉뚱한 결과가 나오기도 하네요.
<div align="center"> <img src="/image/gensim/3.png" /> </div>

 그럼 임베딩이 어떻게 되었나 TSNE를 이용해 2차원, 3차원으로 그려봅니다. 개인적으로 임베딩을 plot 결과가 가장 재밌는 거 같습니다.

 ```
 viz_words = 2000
 word_vector = np.concatenate((model.wv.vectors[:viz_words//2,:], model.wv.vectors[11489:11489+viz_words//2,:]), axis=0)
 tsne = TSNE(n_components=3)
 embed_tsne = tsne.fit_transform(word_vector)

fig, ax = plt.subplots(figsize=(15,15))
for i in range(viz_words):
    if model.wv.index2word[i][-1] == 'N':
        plt.scatter(*embed_tsne[i,:],s=2,alpha=0.6, color='r')
        #plt.annotate(model.wv.index2word[i], (embed_tsne[i,0], embed_tsne[i,1]), alpha=0.6, fontsize=7)
    else:
        plt.scatter(*embed_tsne[i,:],s=2,alpha=0.6,color='b')
        #plt.annotate(model.wv.index2word[i], (embed_tsne[i,0], embed_tsne[i,1]), alpha=0.6, fontsize=7)
plt.savefig("seperate_15000_nolabel.png")
 ```

<div align="center"> <img src="/image/gensim/6.png" /> </div>
<div align="center"> <img src="/image/gensim/7.png" /> </div>
<div align="center"> <img src="/image/gensim/8.png" /> </div>

자세히는 안보이지만 have, has 와 같은 단어들이 뭉쳐있고 숫자들도 따로 뭉쳐있습니다. 파란색을 루머, 빨간색은 루머가 아닌 텍스트에서 나온 단어들로 어느정도 구분이 있는 걸로 보입니다. 300 차원에서는 logistic regression으로 분류가 가능할 것처럼 보이네요.

Gensim의 word2vec은 전처리할 과정도 거의 없어 매우 쉽게 자연어 분석을 할 수 있었습니다. 다음은 doc2vec을 진행하려고 합니다. 문장, 문서를 벡터화하는 모델로 각각의 트위터를 벡터 공간으로 보낼 수 있겠죠.
