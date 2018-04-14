---
title: 텐서플로우를 이용한 Word Embedding
date: 2018-04-13
author: JuHyung Son
layout: post
tags:
  - NLP
  - Word embedding
  - tensorflow
categories:
  - Deep_Learning
---

# Word Embedding

단어를 임베딩하여 벡터 공간으로  옮겨 놓는 것은 여러 측면에서 자연어를 처리하는데 좋은 결과를 보여줍니다. 단지 단어들간의 유사성을 보여주기도 하고 자연어 모델의 성능을 대폭 올려주기도 하죠. 임베딩은 이미지 분야에서 transfer learning을 할 때에 가져오는 모델 파라미터처럼 볼 수 있는데, 텍스트의 특성상 이미지와는 다르게 공통된 특징이 별로 없습니다. 자연어에서 transfer learning과 같은 방법이 많이 안쓰이는 이유죠. 예를 들면, 뉴스 기사 텍스트와 페이스북 댓글 텍스트는 성격이 전혀 다른 텍스트입니다. 각각의 텍스트는 단어의 구조부터 다르고 문체, 문법도 다릅니다. 심지어 소설이라고 해도 작가에 따라 문체, 사용하는 단어가 제각각이기 마련입니다. 그래서 어떤 자연어 프로젝트를 한다면, 자신의 프로젝트에 맞는 텍스트를 모아 자신만의 임베딩 행렬을 만드는 것이 성능에 좋습니다. 이번엔 텐서플로우를 이용해 아주 간단히 공무원법을 임베딩 해봅니다.
- - - -

## 준비물 + 패키지

공무원법과 다음의 패키지들을 사용했습니다. 공무원 법은 1장만 가져왔습니다. 개인 노트북으로 학습해본 것이라 큰 데이터는 사용해보지 못했네요. 클라우드 서버를 사용할 수 있지만 TSNE로 plot하는 과정에서 한글이 깨지는 문제가 해결되지 않아 포기했습니다.
1. 공무원법.txt
2. numpy, tensorflow, utils, collections.Counter, random, matplotlib, sklearn
- - - -

## 과정

자연어는 특히나 전처리에 따라 성능이 확 바뀝니다. 이번엔 아주 기본적인 전처리만으로 진행해봅니다.(사실 아주 기본적인 것도 하지 않았습니다. konlpy 설치에 어려움을 겪는 중..)
가장 처음에 할 과정은 텍스트에 포함된 글이 아닌 것들을 제거하는 과정입니다. 먼저 공무원법 텍스트를 읽어오면 이런 텍스트가 나오죠. konlpy를 사용하면 더 잘할수 있지만 일단 원시적으로 해봅니다.

> ‘국가공무원법\n제1장 총칙 <개정 2008.3.28.>\n\n\n제1조(목적) 이 법은 각급 기관에서 근무하는 모든 국가공무원에게 적용할 인사행정의 근본 기준을 확립하여 그 공정을 기함과 아울러 국가공무원에게 국민 전체의 봉사자로서 행정의 민주적이며 능률적인 운영을 기하게 하는 것을 목적으로 한다.\n[전문개정 2008.3.28.]\n\n제2조(공무원의 구분) ① 국가공무원(이하 "공무원"이라 한다)은 경력직공무원과 특수경력직공무원으로 구분한다.\n② "경력직공무원"이란 실적과 자격에 따라 임용되고 그 신분이 보장되며 평생 동안(근무기간을 정하여...’

\n , ② 같은 비문자?를 없애봅시다.
```
law_word = utils.preprocess(law_text)
```

<div align="center"> <img src="/image/embedding/1.jpg" /> </div>

이런 결과물이 나옵니다. 비문자 텍스트가 오히려 더 생겼습니다. <PERIOD>, 숫자, 공백을 리스트에서 모두 제거합니다.
```
law_word = [c for c in law_word if c not in ['<PERIOD>','<','>','<LEFT_PAREN>','<RIGHT_PAREN>','<QUOTATION_MARK>','<COMMA>']]
remove_list = ['<','>','[',']']
for char in remove_list:
	law_word = [c.replace(char,"") for c in law_word]
law_word = list(filter(None, law_word))
law_word = [item for item in law_word if not item.isdigit()]
```

 <div align="center"> <img src="/image/embedding/2.jpg" /> </div>

이제 어느 정도 정리가 되었습니다. 그런데 한글은 조사를 처리하기가 곤란합니다. ‘공무원의’ 같은 단어는 ‘공무원’으로 보는 것이 임베딩에는 더 좋겠죠. 근데 konlpy를 사용하지 못하고 있으니 그냥 해봅니다.

자연어를 컴퓨터가 처리하기 위해선 각각의 단어를 숫자로 변환시켜줘야 합니다. 이 부분은 utils.create_lookup_tables를 이용하면 편합니다.

```
law_vocab_to_int, law_int_to_vocab = utils.create_lookup_tabels(law_word)
law_int_words = [law_vocab_to_int[word] for word in law_word]
```

<div align="center"> <img src="/image/embedding/3.jpg" /> </div>

이제 조사를 처리하지 못했다는 것만 빼면 단어 처리가 끝났습니다. 근데 위의 단어를 살펴보면 ‘그’, ‘이’,’이다’ 와 같은 애매한 단어들이 있습니다. 노이즈이지만 모두 없애기도 그렇습니다. 이런 단어를 적당히 없애는 게 subsampling 입니다. 단어의 빈도수에 따라 제거하는 방법으로 각 단어를 제거할 확률은 다음처럼 정합니다. $$P(w_i) = 1- \sqrt{ \frac{t}{f(w_i)} } $$
t는 일종의 threshold로 논문에서는 0.00001을 추천했습니다. $f(w_i)$는 단어 $w_i$의 빈도수입니다. 하지만 이 확률은 논문에서 실험을 통해 정해진 확률입니다. 더 좋은 방법이 있다면 찾을 수 있겠지요. 이제 각 단어마다의 제거 확률 리스트를 만듭니다.

```
from collections import Counter
import random

threshold = 1e-5
law_word_counts = Counter(law_int_words)
law_total_count = len(law_int_words)
law_freqs = {word: count/law_total_count for word, count in law_word_counts.items()}
prob = {word: 1-np.sqrt(threshold/law_freqs[word]) for word in law_word_counts}
law_sampled = [word for word in law_int_words if random.random() < (1-prob[word])]
```

다음, 딥러닝 레이어에 넣을 batch를 만듭니다. 저는 skip gram 구조를 이용하려고 합니다. skip gram은 한 단어가 주어지면 주변의 단어를 예측하는 모델이죠. 그럼 인풋은 단어 하나가 되겠고 라벨은  윈도우 사이즈 만큼의 주변 단어가 되겠네요. 이 부분은 함수를 만들어 임베딜 모델안에서 그때그때 생성하는 것이 효율적일 겁니다.

<div align="center"> <img src="/image/embedding/4.jpg" /> </div>

```
#라벨을 만드는 함수
def get_target(words, idx, window_size= 5):
	R = np.random.randint(1, window_size+1)
	temp = idx - R if (idx - R) > 0 else 0
	before = words[temp:idx]
	return before + words[idx+1: idx+R+1]

#인풋, 라벨을 만드는 함수
def get_batches(words, batch_size, window_size = 5):
	n_batches = len(words)//batch_size
	words = words[:n_batches*batch_size)

	for idx in range(0, len(words), batch_size):
		x, y = [], []
		batch = words[idx:idx+batch_size]

		for ii in range(len(batch)):
			batch_x = batch[ii]
			batch_y = get_target(batch, ii, window_size)
			y.extend(batch_y)
			x.extend([batch_x]*len(batch_y))
		yield x, y
```

모든 준비가 끝났습니다! 임베딩 모델을 만듭시다. 임베딩 모델은 아주 간단합니다. 인풋, 히든, 아웃풋 레이어로 되어 있죠. 크지 않은 데이터라면 cpu만으로도 조금은? 가능합니다. 이 과정에서 중요한 건, Negative sampling입니다. 이것 덕분에 학습을 cpu에서 더 빠르게 할 수 있죠. 텐서플로우에는 다행히 negative sampling이 `tf.nn.sampled_softmax_loss`로 구현되어 있습니다.
```
#인풋 레이어를 만듭니다.
train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, [None], name='inputs')
    labels = tf.placeholder(tf.int32, [None, None], name='labels')

#임베딩 레이어를 만듭니다.
n_vocab = len(law_int_to_vocab)
n_embedding =  200 # Number of embedding features
with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs)

#Negative sampling
n_sampled = 100
with train_graph.as_default():
    softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding), stddev=0.05))# create softmax weight matrix here
    softmax_b = tf.Variable(tf.zeros(n_vocab))# create softmax biases here

    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b,
                                      labels, embed,
                                      n_sampled,n_vocab)

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
```

레이어는 모두 구축하였고 학습을 시켜봅니다.

```
epochs = 500
batch_size = 32
window_size = 10

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs+1):
        batches = get_batches(law_sampled, batch_size, window_size)
        start = time.time()
        for x, y in batches:

            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss

            if iteration % 10 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss/100),
                      "{:.4f} sec/batch".format((end-start)/100))
                loss = 0
                start = time.time()


            iteration += 1
    embed_mat = sess.run(normalized_embedding)
```

<div align="center"> <img src="/image/embedding/5.jpg" /> </div>

이렇게 임베딩을 시켜보았습니다. 500 에폭을 돌렸는데 역시 거의 전혀 학습이 되질 않습니다. 조사를 처리하지 않았고 일단 데이터의 양이 부족합니다. 일단 konlpy를 설치하고 한글 텍스트 전처리를 어느정도 해본 후 다시 해봐야겠네요.
<hr>
해봤습니다. 일단 konlpy에 kkma 클래스를 사용해서 전처리를 해보았습니다. konlpy에 여러 클래스가 있는데 표를 보니 Mecab이 더 나은 거 같아 써보려 했지만 뭐가 문젠지 Mecab이 설치되지 않는... 일단 kkma.morphs를 한 결과는 이전보다 훨씬 좋습니다. 또 이번엔 re 를 사용해 전처리를 해서 좀 더 깔끔한 데이터가 만들어졌네요.

<div align="center"> <img src="/image/embedding/6.jpg" /> </div>

장, 이, 은 과 같은 단어는 subsampling이 어느 정도 해결해 줄 것이라 믿고 해봤습니다. 그런데 일단 역시 데이터가 너무 작아서인지 잘 안되는 듯 보이고 학습이 거의 되질 않습니다. 좀 더 큰 데이터 셋을 마련한 후 서버에서 환경 맞춘 후 다시 시도해보겠습니다.
<div align="center"> <img src="/image/embedding/7.jpg" /> </div>
