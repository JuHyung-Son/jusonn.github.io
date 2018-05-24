---
title: 문서 분류를 위한 CNN 모델
date: 2018-05-18
author: JuHyung Son
layout: post
tags:
  - NLP
categories:
  - Deep_Learning
---

문서 분류는 자연어 처리에서 가장 많이 다루는 주제 중 하나입니다. 감성 분석, 영화 리뷰 분석, 스팸 메일 분류 등 여러 곳에 쓰이죠. 스팸 메일 분류와 같은 건 기존의 나이브-베이즈 모델도 굉장히 잘 작동하지만 나머지 다른 곳에선 쓸만한 성능을 보이지 못하기도 합니다. 역시 이런 문제들에도 딥러닝을 적용한 연구들이 진행 중이고 state-of-art 정도의 결과를 내며 잘 작동하기도 합니다. 특히 CNN을 이용한 모델이 잘 작동하는데요. 이미지에서만 쓰이던 CNN이 자연어처리에 쓰인다니 좀 어리둥절 했습니다.

이번에는 Embedding + CNN을 이용한 문서 분류 모델을 보겠습니다.

## Word Embedding + CNN = Text Classification

CNN을 이용한 이미지 분류에서 CNN은 이미지의 두드러진 특징들을 굉장히 잘 뽑아냅니다. Convolutional 레이어를 간단히 설명하면 이미지에서 의미있는 특징만을 추출해내는 것이죠. 그런데 이런 특징은 텍스트에서도 적용됩니다. CNN은 기본적으로 인풋이 이미지, 즉 2D 혹은 3D 라고 가정하고 만들어진 모델이기 때문에 어떻게 텍스트를 인풋으로 넣을 수 있지 하는 의문이 들지만, 간단하게 kernel와 pooling 과정을 2D가 아닌 1D로 진행해주면서 이것이 가능하게 됩니다.

CNN으로 텍스트를 분류하는데 가장 핵심적인 과정은 다음과 같습니다. 기본적으로 CNN 모델이니 convolution, fc 레이어가 있고 텍스트를 가장 잘 표현하기 위해서 임베딩을 사용하죠.

- Word Embedding
- Convolutional model
- Fully connected model

임베딩은 텍스트를 표현하는데 아주 효과적인 방법입니다. 문맥에서 나타나는 단어들간의 의미를 나타내기도 하고 번거로운 단어처리도 덜어주죠. 하지만 좋은 성능의 단어 벡터를 얻는데에는 아주 많은 텍스트와 학습시간이 걸립니다. 그래서 연구자들이 제공하는 학습된 단어벡터를 쓰는게 일반적이죠.

단어를 벡터로 표현하여 한 단어가 약 200차원의 벡터로 표현된다고 해봅니다. 그렇다면 m개의 단어를 가진 한 문장은 m*200 인 행렬로 표현되죠. 여기에 convolution 레이어를 적용해줍니다. 이렇게 보면 사실 이미지 분류에서의 과정이랑 상당히 비슷해 보입니다.

<div align="center"> <img src="/image/cnndoc/1.png" /> </div>

<div align="center"> Convolutional Neural Networks for Sentence Classification </div>

## Sentiment analysis with CNN

CNN 모델을 적용하여 트위터가 루머인지 아닌지 분류하는 모델을 만들어 봅시다.

데이터는 Harvard Dataverse 에서 제공하는 트위터 데이터로 루머인 트윗과 루머가 아닌 트윗으로 나누어져 있고 110개의 주제로 분류되어 있습니다. 사실 데이터가 편향되어 있어 이 데이터에 대해 학습을 하는 것이 무의미합니다. 그렇지만 모델을 직접 구현해보는 것에 초점을 맞추고 진행합니다.

<div align="center"> <img src="/image/cnndoc/2.png" /> </div>

데이터는 이렇게 한 주제에 관한 트윗들이 있습니다. 그리고 그 한 주제가 루머인지 아닌지 라벨이 되어있죠. 이 데이터가 아쉬운 점은 한 주제가 루머, 비 루머 트윗을 모두 포함하고 있지 않다는 것입니다. 예를 들어 Airfrance에 대한 트윗은 루머 트윗만이 있죠. 이런 데이터의 구성 때문에 간단한 모델로 성능을 보이긴 어렵습니다.

<div align="center"> <img src="/image/cnndoc/3.png" /> </div>

그리고 각 주제에 대해 이렇게 트윗들이 나열되어 있습니다.

가장 먼저 텍스트를 전처리를 통해 딥러닝 모델에 넣을 수 있게 만듭니다. 전처리 파이프라인은 먼저 텍스트를 단어들로 쪼갠 후 큰 리스트에 담아봅니다.

```python
from nltk.corpus import stopwords

def get_tokens(txt):
  tokens = txt.split()
  tokens = [word for word in tokens if word.isalpa()]
  stop_words = set(stopwords.words('english'))
  tokens = [w for w in tokens if not w in stop_words]
  tokens = [w for w in tokens if len(word) < 15]
  tokens = [w for w in tokens if len(word) > 1]
  return tokens

non_rumor = list()
rumor = list()

for line in ntweets:
  tokens = get_tokens(line)
  non.append(tokens)
for line in rtweets:
  tokens = get_tokens(line)
  rumor.append(tokens)

total = non + rumor
labels = np.array([0 for _ in range(len(non))] + [1 for _ in range(len(rumor))])
```

다음 ```from keras.preprocessing.text import Tokenizer``` 를 이용해 bow를 얻고 문장에 패딩을 주어 같은 사이즈의 리스트로 만들어 줍니다. 이렇게 일단 모델에 넣을 데이터 처리가 끝납니다.

```python
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def encode_docs(tokenizer, max_length, docs):
    encoded = tokenizer.texts_to_sequences(docs)
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded

x_train, x_test, y_train, y_test = train_test_split(total, labels, test_size=0.3)
tokenizer = create_tokenizer(total)
vocab_size = len(tokenizer.word_index) + 1
max_length = max([len(s.split()) for s in total_tweets if s is not None])

x_train = encode_docs(tokenizer, max_length, x_train)
x_test = encode_docs(tokenizer, max_length, x_test)
```

다음 CNN 모델을 아주 간단하게 만들어 봅니다.

```python
def cnn_model(tokenizer, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
```

pre-trained word vector 를 사용하는 것이 성능에 좋지만, 먼저 keras의 Embedding을 사용합니다.
1차원 convolution 레이어를 사용하고 필터는 32개 커널 크기는 8개를 선택했습니다.

<div align="center"> <img src="/image/cnndoc/4.png" /> </div>

한번 학습을 해보면..

<div align="center"> <img src="/image/cnndoc/5.png" /> </div>

그냥 딱봐도 오버피팅이 일어난 거 같습니다.
이 현상은 사실 데이터 자체가 좋은 데이터가 아니기 때문에 어쩔 수 없이 일어나는 현상입니다. 위에서 말했듯이 데이터에는 Airfrance와 관련된 루머 트윗은 있지만 루머가 아닌 트윗은 없죠.

즉 모델은 Airfrance라는 단어가 있는 트윗은 모두 루머라고 판단하게 되는 것이죠. 학습 데이터와 검증 데이터, 테스트 데이터 모두 데이터가 편중되어 있기 때문에 정확도가 100%에 가깝게 나오고 있고요. 다음과 같은 문장을 한번 모델에 넣어보겠습니다.

> Obama killed earwigs

여기서 obama에 관련된 트윗은 루머, 비루머가 모두 존재하지만 earwigs에 관련된 트윈은 비 루머만이 존재합니다. 그렇다면 모델은 데이터를 학습하면서 earwigs 가 들어간 문장은 모두 비 루머라고 학습을 했겠죠.

<div align="center"> <img src="/image/cnndoc/7.png" /> </div>

이렇게 말입니다.. 성능은 좋지 않지만 일단 CNN으로도 문서 분류를 해보았습니다.
