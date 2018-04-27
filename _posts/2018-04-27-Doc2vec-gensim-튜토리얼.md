---
title: Distributed representations of Sentences and Documents 그리고 gensim
author: JuHyung Son
layout: post
tags:
   - NLP
   - python
   - doc2vec
   - paper
categories:
  - Paper
---

# Distributed representations of Sentences and Documents

이번 논문은 doc2vec 이라고 불리는 모델을 개발한 논문 소개입니다. 이 논문 역시 구글에서 나온 논문으로 제목에서 알 수 있듯이, 문장과 문서를 벡터로 표현하는 모델을 제시하는 논문입니다. 논문에서는 sentences라는 표현보다는 input sequences of variable length 라는 표현으로 좀 더 넓은 범주의 문장을 표현하면서 텍스트 뿐만이 아닌 다른 sequence 데이터에도 적용이 가능할 것이라고 합니다.

## Intro

기존에 자연어 처리에서 가장 많이 쓰인 모델 중 하나인 Bag of words는 꽤 치명적인 약점이 있습니다. 먼저는 단어의 순서를 고려하지 않는다는 것으로 언어에서 순서가 꽤나 중요하단 걸 생각하면 이것만으로도 꽤나 큰 약점이죠. 또 다른 하나는 단어의 의미적인 구조를 고려하지 않는다는 것입니다. "Powerful", "Strong", "Paris" 이 단어중 "Powerful"과 "Strong"은 의미적으로 비슷함에도 단어들 간의 연관성이 고려되지 않는다는 것이죠.

이 두 약점은 word2vec 모델이 나오면서 보완이 되었습니다. 그리고 저자는 wor2vec에서 영감을 받아 단어만 벡터로 표현하는 것이 아닌, variable-length의 인풋을 벡터로 표현하고자 합니다. Variable-length of input은 결국 문장, 문단이 되겠죠.

## Paragraph Vector

문장을 벡터화하는 건 생각보다 엄청나게 단순합니다. word2vec을 알고 있다면요. doc2vec은 word2vec의 확장판이라고 봐도 무방할 정도입니다. 그리고 이 때문인지, Word2vec의 장점이 그대로 남아있기도 합니다. 저자는 paragraph vector라는 것을 제시합니다. 이 paragraph vector는 단지 word2vec 모델에 paragraph matrix를 추가한 것입니다. 기존의 word vector W에 paragraph vector인 D를 추가한거죠. word2vec에서의 W가 W+D가 된 것이 유일한 변화입니다.

<div align='center'> <img src="/image/gensim/doc2vec/1.png" />  </div>

Word2vec에서의 목표는 아래 로그 확률의 평균을 최대화 시키는 것입니다.

$$\frac{1}{T} \sum^{T-k}_{t=k} log p(w_{t} | w_{t-k},...,w_{t+k}) $$

그리고 단어를 예측하는 과정은 softmax를 사용해 나타냅니다.

$$p(w_{t} | w_{t-k},...,w_{t+k}) = \frac{e^{y_{w_i}}}{\sum_{i} e^{y_i}}$$

그리고 라벨인 y는 다음 처럼 계산합니다.

$$y = b+Uh(w_{t-k},...,w_{t+k};W)$$

Doc2vec은 마지막의 y식을 다음과 같이 변환한 것입니다.

$$y = b+Uh(w_{t-k},...,w_{t+k};W, D)$$

위와 같이 기존의 word vector에 paragraph vector를 추가해 준 것을 Distributed memory model 이라고 합니다. 저자에 따르면 이 paragraph 토큰이 메모리와 같은 기능으로 현재의 문맥에서 빠진 것이나 단락에서의 주제를 기억해주는 역할을 한다네요. 그런 이유로 이 모델을 Distributed Memory Model of Paragraph Vectors(PV-DM)이라고 한답니다. paragraph 토큰은 정확하진 않지만 아마 문장과 문단의 주제, 태그를 의미하는 것 같습니다.

반면 word vector를 빼고 paragraph vector 만으로 학습을 하는 모델을 Distributed bag of words (PV-DBOW)라고 합니다.
<div align='center'> <img src="/image/gensim/doc2vec/2.png" />  </div>

논문의 실험에서 PV-DM의 성능이 거의 대부분 좋게 나와 PV-DBOW는 많이 다루지 않은 듯합니다. 저자는 PV-DM과 PV-DBOW 둘을 혼합한 모델을 강력히 추천한다고 하네요.


## Gensim

그럼 gensim으로 튜토리얼을 해봅니다. 데이터는 이전에 처리해 놓은 트위터 데이터 주제가 분류되어 있는 텍스트입니다.

<div align='center'> <img src="/image/gensim/doc2vec/3.png" />  </div>

데이터의 content 트윗을 한 인풋으로 하여 doc2vec을 해봅니다. gensim의 doc2vec은 모듈이 원하는 인풋 형태를 따로 처리해주어야 합니다. 제 경우 데이터프레임 안에 content만 불러와 tag를 씌웁니다. 일단 태그는 topic과 같게 주었는데 맞는 방법인지 모르겠습니다. `topic_1` 처럼 인덱스를 주는 게 일반적으로 보입니다.
```python
def read_corpus(dataframe, tokens_only=False):
    for i in range(len(dataframe)):
        try:
            if tokens_only:
                yield gensim.utils.simple_preprocess(dataframe.content[i])
            else:
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(dataframe.content[i]), [dataframe.topic[i]])
        except:
            pass

N_corpus = list(read_corpus(N_tweets))
R_coupus = list(read_courpus(R_tweets))
```
태그를 씌우면 doc2vec에 학습할 수 있는 형태로 됩니다.
<div align='center'> <img src="/image/gensim/doc2vec/4.png" />  </div>

이제 두 줄의 코드로 doc2vec을 학습할 수 있습니다. 말뭉치를, PV-DM 모델로, 200차원으로, 최소 단어는 3개로 10번 그리고 negative sampling을 사용해여 학습합니다.
```python
model = gensim.models.doc2vec.Doc2Vec(N_corpus+R_corpus, dm=1, vector_size=200, min_count=3, epochs=10, hs=0)
model.train(N_corpus+R_corpus, total_examples=model.corpus_count, epochs=model.epochs)
```
학습을 마친 후 `N_Airfrance`라는 주제와 가장 비슷한 주제를 찾아봅시다.
```python
model.doc2vec.most_similar('N_Airfrance')
```
<div align='center'> <img src="/image/gensim/doc2vec/5.png" />  </div>

잘 된건지는 모르겠습니다. 단어 벡터는 잘 학습되었나 봅니다.

```python
model.wv.most_similar('great')
```
<div align='center'> <img src="/image/gensim/doc2vec/6.png" />  </div>

단어벡터는 학습이 잘 된거 같군요. 하지만 함정이 하나 보이긴 합니다.. terrible이 wonderful보다 높군요..?
이번엔 아무 문장이나 만들어서 그것이 어떤 주제와 가까운지 봐봅니다.

```python
inferred = model.infer_vector('obama is elected as a US president again!')
sims = model.docvecs.most_similar([inferred])
sims
```

<div align='center'> <img src="/image/gensim/doc2vec/7.png" />  </div>

역시 잘 되지는 않습니다. 그리고 코드를 반복할 때 마다 다른 결과가 나타나곤 합니다. 내부에서 어떤 과정이 일어나는지 좀 더 봐야할 부분이네요.
