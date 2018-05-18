---
title: 추천 시스템 들어가기 파트2 (Neural Network Approach)
date: 2018-05-18
author: JuHyung Son
layout: post
tags:
  - Recommender
categories:
  - Studying
---

Steeve Huang 의 Introduction to Recommender System. Part 2 (Neural Network Approach) 를 번역한 것입니다.

<a href="https://towardsdatascience.com/introduction-to-recommender-system-part-2-adoption-of-neural-network-831972c4cbf7"> 원문보기 </a>

# 1. Introduction

저번 포스팅에서는 협업 필터링 (Collaborative Filtering) 과 차원 축소 (Singular Value Decomposition)가 어떻게 추천 시스템에 쓰이는 지 보았습니다. 인공 신경망이 대세인 요즘, 신경망 기법들이 어떻게 추천 시스템에 적용되는 지도 궁금하죠. 이번 포스팅에서는 PyTorch 기반의 추천 시스템 프레임 워크인 Spotlight와 Item2vec을 보겠습니다.

# 2. Spotlight

<div aligh="center"> <img src="/image/recommender/2.png" /> </div>

Spotlight는 추천 시스템을 만드는데 쓰이는 파이썬 프레임 워크입니다. 크게 두가지 종류의 모델이 들어있는데 바로 factorization 모델과 sequence 모델입니다. Factorization 모델은 SVD와 같이 utility matrix를 유저와 아이템에 관한 latent 형태로 분해한 후 그것은 네트워크에 넣는 아이디어를 사용합니다. Sequence 모델은 LSTM이나 1차원 CNN 같은 시계열 모델과 함께 만들어집니다. Spotlight는 pytorch 기반이므로 설치하기 전에 PyTorch가 설치되어 있는지 확인해야 합니다.

## Interactions

Spotlight 에선 utility matrix를 Interactions 라고 부릅니다. Implicit Interaction 를 만들기 위해서 먼저 유저와 아이템의 ID 를 만들어야합니다. 여기에 레이팅 정보를 추가하면 explicit Interaction 이 되죠.

```python
from spotlight.interactions import interactions

implicit_interactions = Interactions(user_ids, item_ids)
explicit_interactions = Interactions(user_ids, item_ids, ratings)
```
## Factorization Model

Factorization Model 은 implicit 혹은 explicit Interaction을 이용합니다. 먼저 상대적으로 간단한 implicit 모델을 봅니다.

$$I = \begin{pmatrix} 1.0 & 0.0 & ... & 1.0 \\ 0.0 & 0.0 & ... & 1.0 \\ ... & ... & ... & ... \\ 1.0 & 1.0 & ... & 1.0 \end{pmatrix}$$

이 아이디어는 기존의 SVD 와 매우 유사합니다. 유저와 아이템을 latent 공간으로 옮겨 직접적으로 비교 가능하게 하는 것이죠. 본질적으로는, 유저와 아이템을 나타내기 위해 각각의 임베딩 레이어를 사용합니다. 이때 타겟은 인풋으로 들어가는 interaction (utility matrix) dlqslek. 유저-아이템 쌍의 점수를 계산하기 위해, 유저와 아이템 각각의 latent로 타낸 행렬의 내적을 구합니다.

<div align="center"> <img src="/image/recommender/3.png"/> </div>

Spotlight 에선 몇 줄의 코드로 모델을 학습할 수 있습니다. Sci-kit learn과 매우 비슷하죠.

```python
from spotlight.factorization.implicit import ImplicitFactorizationModel

implicit_model = ImplicitFactorizationModel()
implicit_model.fit(implicit_interactions)
implicit_model.predict(user_ids, item_ids=None)
```

## Sequential Model

Sequential model 은 추천 문제를 시계열의 성격을 띄는 예측 문제로 바꿉니다. 과거의 interaction이 주어지면 다음 시간에 한 유저가 가장 좋아할 상품을 찾는 것입니다. 예를 들어, 유저 A의 순서로 나타낸 아이템 [2, 4, 17, 3, 5] 에 대한 interaction 을 갖고 있다고 합니다. 그러면 다음과 같은 window 예측 모델이 나옵니다.

```python
[2] -> 4
[2, 4] -> 17
[2, 4, 17] -> 3
[2, 4, 17, 3] -> 5
```

왼쪽의 배열은 과거의 interaction 을 나타내며 오른쪽의 숫자는 A가 다음에 고를 아이템입니다.

이러한 모델을 학습하기 위해 위에 있던 interaction 을 순서가 있는 형태의 interaction 으로 변환해야 합니다. 단지 이것만 추가하면 되죠.

```python
from spotlight.sequence.implicit import ImplicitFactorizationModel

sequential_interaction = implicit_interactions.to_sequence()
implicit_sequence_model = ImplicitSequenceModel()
implicit_sequence_model.fit(sequential_interaction)
```

위 코드에 `to_sequence()` 는 zero 패딩이 앞단에 적용됩니다. 아래와 같이요.

```python
[0, 0, 0, 0] -> a
[0, 0, 0, a] -> b
[0, 0, a, b] -> c
[0, a, b, c] -> d
```

그러므로 id가 0인 아이템은 0이 아닌 다른 id 숫자로 바꿔야 합니다.

## Choice of Loss Function

모델을 만들 때, 비교적 여러 선택지의 비용함수가 있습니다. 비용함수만 바꿔주어도 모델은 완전히 다른 성능을 보이죠. Spotlight에서 쓰이는 주요 비용함수는 2가지입니다.

- 'pointwise': 이건 가장 간단한 형태의 비용함수입니다. 하지만 샘플의 sparsity 때문에 모든 아이템에 대해 계산할 수는 없습니다. 그래서 주어진 유저에 대해 모든 아이템의 비용함수를 구하기보다는, 무작위로 선택된 negative 샘플과 positive 샘플의 일부분만을 사용합니다.
- 'bpr': Bayesian Personalized Ranking(BPR) 은 각 유저에 대해 모든 아이템에 순위를 매깁니다. 이 때 다음 식을 이용해 positive 샘플의 순위는 negative 샘플보다 높게 만듭니다.

$$L = 1.0 - sigmoid (positive - negative)$$

<div align="center"> ranking loss </div>

이렇게 Spotlight를 이용해 추천 시스템을 구축합니다. 당신이 필요한 모델을 유연성있고 쉽게 구축할 수 있져. Sequence 모델이 factorization 모델보다 좋은 성능을 보이긴 하지만 학습하는데에 더 많은 시간이 걸립니다. 게다가, sequence 모델을 적용할 때 데이터 자체에 sequential 한 상관관계가 없다면 성능을 내기도 힘듭니다.

# 3. Item2Vec

Item2vec 은 지난달 IDAO 대회에서 생각난 아이디어입니다. 대회는 Yandex를 위한 추천 시스템을 만드는 대회였습니다. 당시 저는 Word2vec을 공부하고 있었고요. 추천 시스템에도 이와 비슷한 아이디어가 적용될 수 있을 거라는 생각이 들었습니다. 이 아이디어가 있는 논문이 있는지는 확실치 않았지만, 추천 분야에서 word2vec의 응용이 쓰인 것은 본 적이 없었습니다. Word2vec 의 간단한 아이디어는 긱각의 단어를 분산 형태로 나타내는 것입니다. 즉 각 단어는 그 단어 주위의 단어들로 결정되는 벡터로 표현됩니다. 비슷하게, 유저와 관계가 있는 아이템을 분산 형태로 나타내는게 제 생각이었습니다.

각 유저에 대해 먼저 시간 순서로 정렬된 아이템 리스트를 만들었습니다. 그리고나서 Gensim의 word2vec 모델을 이용해 이 아이템 리스트를 학습하였습니다. 그 다음 모델을 저장했습니다.

```python
from gensim.models import Word2Word2Vec

model = Word2Vec(item_list, size = 50, window = 5, min_count = 5, workers = 10, sg = 0)
model.wv.save_word2vec_foramt("data/item_vectors.txt")
```

다음 학습된 아이템을 임베딩 행렬에 넣습니다.

```python
item_index = {str(i):i for i in range(931)}
embeddings_index = {}
f = open("data/item_vectors.txt")

for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((max_features, embed_size))

for word, i in item_index.items():
  if i < max_features:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
```

이제 유저의 다음 interaction 을 예측하는 모델을 만듭니다. 기본적으로 GRU 모델이 쓰입니다.

```python
maxlen = 75
output_size = y_train.shape[1]
max_features = output_size
embed_size = 50
input_size = (maxlen, 1, )

def get_model():
  global input_size, output_size
  inp = Input(shape=input_size)
  x = Embedding(max_features, embed_size)(inp)
  x = CuDNNGRU(50, return_sequences=True)(inp)
  x = LeakyReLU()(x)
  x = Dropout(0.2)(x)
  x = GlobalMaxPolling1D()(x)
  x = Dense(40)(x)
  x = Dropout(0.2)(x)
  x = Dense(output_size, activation="sigmoid")(x)
  model = Model(inputs=inp, outputs=x)
  model.compile(loss="categorical_crossentropy", optimizer='rmsprop')
  model.layers[1].set_weights([embedding_matrix])
  model.layers[1].trainable = False
  return model
```

미리 학습 시켜놓은 임베딩 행렬은 학습하지 않도록 해놓아야 합니다.

# 4. Performance

이 모델은 IDAO 대회의 데이터 셋에 적용시켜 성능을 보았습니다.

|Model | Score|
|:--:|:--:|
| SVD | 2218  |
|Spotlight Implicit Factorization   | 1223  |
|Spotlight Explicit Factorization   | 1337  |
|Spotlight Sequence   | 2221  |
|Item2Vec   | 2492  |

인공신경망 모델이 꼭 기존의 추천 시스템 모델보다 좋은 성능을 보이진 않습니다. SVD 모델은 학습시간이 훨씬 오래 걸리는 Spotlight Sequence 모델과 비슷한 성능을 보이죠. Item2Vec 은 놀랍게도 가장 좋은 모델로 보입니다. 단지 한번의 실험으로 판단할 순 없지만 각 모델에 대한 대략적인 직관을 얻을 수 있습니다.

# 5. Conclusion

Spotlight에 내장된 두가지 모델과 Item2Vec 을 보았습니다. 또 IDAO에서 제공한 데이터를 가지고 여러 모델의 성능도 보았습니다. SVD가 가장 효율적인 모델로 보여지고 Item2Vec 도 추천 시스템에 그 가능성을 보여주었습니다.
