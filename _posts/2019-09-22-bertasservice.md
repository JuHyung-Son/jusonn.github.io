---
title: RERT AS SERVICE 블로그 글 번역
date: 2019-09-22
author: JuHyung Son
layout: post
categories:
  - Dev
  - Deep_learning
---
hanxiao의 [bert-as-service](https://hanxiao.github.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/)를 번역하였습니다.

#  BERT as service with tf and ZeroMQ

- BERT와 ZeroMQ 기반의 스케일러블한 센텐스 엔코딩 오픈소스 프로젝트 bert as service 의 디자인 철학, 구조에 관한 글이다. 이 프로젝트는 한 문장의 고저된 길이의 벡터로 변환하는 프로젝트이다.

## 백그라운드

2018년 ML과 NLP 에서 가장 큰 뉴스는 구글의 BERT라고 불리는 Bidirectional Encoder Representations from Transformers 였다. BERT는 언어의 representation을 pre-training하는 방법을 사용하였는데 거의 모든 NLP 대회에서 신기록을 세웠다.

텐센트 인공지능 랩은 BERT가 함축적인 텍스트를 고정된 길이의 벡터로 변환하는 새로운 방법에 흥미가 있었다. 그들이 하고 있는 많은 NLP/AI 어플리케이션에서 효과적인 벡터 표현은 아주 중요했기 때문이다. 예를 들어 딥러닝 기반의 정보 검색을 보면, 쿼리와 도큐먼트는 같은 벡터 공간으로 매핑된 다음 둘 간의 연관성을 유클리디언 혹은 코사인 거리 같은 적절한 메트릭을 통해 계산한다. 벡터 표현이 효과는 검색의 질과 바로 연결된다.

그래서 많은 NLP 팀들이 semantic feature(벡터 표현과 같은)에 의지한다면, 다수의 팀들에게 서빙을 제공할 수 있는 문장 인코딩 인프라를 제공하면 좋지 않을까? 이 아이디어는 매우 간단하고 직관적이지만 최근까지도 실용적인 방법은 아니었다. 왜냐면 많은 딥러닝 알고리즘들이 특정 도메인이나 작업에 치우친 벡터 표현을 제공했기 때문이다. 결과적으로 한 팀에서 얻은 벡터 표현은 다른 팀에게는 쓸모없는 표현일 수도 있는 것이다. 반면에 BERT, ELMo, ULMFit 과 같은 모델은 NLP 과정을 작업과는 독립적인 pre-training과 작업에 연관된 fine-tuning으로 분해한다. pre-training 과정에선 NLP에 대해 일반적이고 다른 태스크에도 쓸 수 있는 모델을 학습하게 된다. 

지난 몇 주 동안, BERT를 이용해 문장 인코딩 서비스를 하는 bert-as-service라는 오픈소스 프로젝트를 구현했다. 서빙 속도, 적은 메모리 사용과 scalability에 특화된 서비스다. README.md에 사용법이 잘 나와있다. 여기서는 기술적인 사항, 특히 이 프로젝트에 디자인에 관해 이야기한다. 당신도 텐서플로 모델을 프로덕션에 서빙한다면 참고가 될 것이다.

## BERT Recap

BERT의 대표적인 특징은 multi-head self-attention network, dual training task, 대규모 TPU 학습이 있다. 이런 특징들로 모든 NLP 대회를 싹쓸이했다. 하지만 좋은 퍼포먼스는 단지 BERT가 가진 하나의 면이다. 내가 가장 좋아하는 건 BERT의 디자인 패턴이다. BERT는 일반적인 목적에 사용 가능한 "언어 이해" 모델을 큰 말뭉치를 이용해 학습한다. 그리고 그걸 다양한 분야에 사용할 수 있는 모델에 사용한다. End-to-end training과 다르게 BERT는 전통적인 NLP 를 두가지로 나누었다: pre-training, fine-tuning 으로.

- Pre-training은 일반적인 텍스트 데이터를 문법, 컨텐스트 정보의 이해를 위해 학습한다.
- Fine-tuning은 특정한 작업, 도메인이나 데이터로부터 학습한다.

<div aligh="center"> <img src="/image/bertasservice/1.png" /> </div>

그러나 pretraining은 많은 자원과 오랜 시간이 걸리는 작업이다. 구글에 따르면 pretraining을 끝내기 위해 4~16 TPU로 4일이 걸렸다고 한다. 연구자와 엔지니어가 인내할 수 없는 부분이다. 다행스럽게도 이 과정은 모든 언어에 대해 딱 한번만 하면 되고 그걸 구글이 공개했다. 링크. pretrained BERT를 다운받고 모델로부터 나온 아웃풋을 모델에 넣어주기만 하면 된다. 제너럴한 언어 정보가 이미 pretrained BERT에 담겨있으므로, 가벼운 사이즈의 모델을 사용하는게 효과적이다. 또한 당연히 모든 네트워크 (BERT를 포함한)을 fine-tuning 할수도 있다.

## BERT as a service

우리가 BERT로부터 얻고 싶은건 뭘까? IR/search 도메인에서는 쿼리와 문서의 벡터 표현을 얻고 코사인 유사도를 이용해 둘의 차이를 계산하고 싶어한다. 분류 작업에서는 벡터 표현과 더불어 분류기를 학습시킬 라벨을 얻고 싶어한다. 앙상블 학습 시스템에서는 BERT를 통해 얻은 벡터를 피처의 한 부분으로 갖고 싶어한다. 공통적으로 텍스트를 주면, 그것을 표현하는 벡터를 얻고 싶어하는 것이다. 이것이 많은 AI 팀의 요구사항 이었다.

그럼 BERT 서비스로부터는 뭐가 필요한가? 빠른 추론 속도, 적은 메모리 사용과 높은 확장성이었다. 이건 서비스 제공자의 기본적인 요구사항이다. 고객의 입장에선 서비스가 사용하기 쉬워야한다. API는 `vector = encode(texts)` 와 같이 직관적이어야 한다.

<div aligh="center"> <img src="/image/bertasservice/2.png" /> </div>

## Research: Finding an Effective Embedding/Encoding

직접 해보기 전에, 어떻게 BERT로부터 효과적인 문장 인코딩을 얻을 수 있을지 보자. BERT 모델은 12/24 레이어를 가지고 각각은 이전의 레이어로부터 self-attends 하고 `batch_size, seq_length, num_hidden` 의 아웃풋을 가진다. 단어 임베딩을 얻는건 직관적이다. 하지만 문장 임베딩을 얻을 때는 `batch x tokens x dimenstion` 텐서를 `batch x dimension` 크기로 풀링해야한다. 여기에는 average pooling, max pooling, hierarchical pooling 등 여러 방법이 사용된다. 물론 풀링과정은 새 파라미터를 사용하는 것이 아닌 단순한 연산 과정이다.

많은 풀링 방법이 있지만, 구글의 BERT 논문을 보면 두 특별한 토큰 [CLS], [SEP]이 인풋 문장의 시작과 끝에 패딩으로 각각 들어가 있다. 한번 downstream 작업이 fine-tune되면, 앞의 두 토큰의 임베딩은 전체 문장을 표현할 수 있게 된다. 사실, 이건 공식 BERT 소스 코드에서 사용된 방법이다. 그러므로, 이것을 사용하면 된다. 하지만, 만약 BERT가 단지 pretrain 되었고 downstream 작업에 fine tune 되지 않았다면, 저 두 토큰의 임베딩은 의미가 없다.

우리는 여러 풀링 전략을 갖고 있는데 이제 어떤 레이어를 이걸 적용해야 할까? 사람들은 경험적으로 마지막 레이어를 사용할 것이다. 그러나 BERT는 두 부분의 타겟(masked language model, next sentence prediction)으로 학습되었다는 걸 기억해야 한다. 마지막 레이어는 이 타겟을 위해 학습된 레이어라서 이 두 타겟에 편향되어 있는 상태이다. 일반화의 의미에서 우리는 마지막에서 두번째 레이어로 풀링을 쓰는것이 좋다. BERT 레이어들은 서로 다른 정보를 뽑아낸다. 이것을 명확하게 보려고 UCI-News Aggregator Dataset과 pretrained uncased_L-12_H-768_A-12 를 이용해 시각화를 해보았다. 무작위로 20K의 기사 제목을 뽑아 서로 다른 레이어에서 인코딩하고 max, average 풀링을 하였다. 그리고 PCA를 통해 2차원으로 나타내었다. 데이터에는 오직 4개의 클래스가 있고 각각을 색깔로 나타내었다.

<div aligh="center"> <img src="/image/bertasservice/3.png" /> </div>

비슷한 층의 레이어는 비슷한 표현을 뽑아낸다는게 보인다. 반면, 처음 몇 레이어와 마지막 몇 레이어는 상당히 다른 의미를 표현하는 듯 하다. 가장 깊은 레이어는 아마 단어의 원래 의미를 보존할 것이다. 그래서 단순한 워드 임베딩과 비슷한 성능을 보일 것이다. 그러므로 첫 레이어와 마지막 레이어 사이에서 trade-off를 해야한다.

## 엔지니어링: 확장가능한 서비스 만들기

다행히 풀링이 유일하게 연구해야 할 문제였다. 다른 부분은 엔지니어링에 더 초점이 맞춰 있었다.

- BERT와 downstream 모델을 분리

가장 먼저 우리는 BERT와 downstream 모델을 분리해야 했다. 구체적으로, 12/24개의 multi-head attention 레이어는 다른 프로세스 혹은 다른 머신에서 돌아야 했다. 예를 들어, cost-per-use GPU 머신에서 이걸하고, 동시에 여러 팀에 서빙하는 것이다. downstream 모델은 주로 가볍고 딥러닝 라이브러리조차 필요없는 경우가 많아 CPU 머신이나 모바일에서 돌 수 있었다.

<div aligh="center"> <img src="/image/bertasservice/4.png" /> </div>

두 역할의 분리는 client/server 역할을 더 명확히 한다. 피쳐 추출이 보틀넥 일 때는 GPU 서버를 확장하면 된다. downstream 모델이 보틀넥 일 때는 CPU 머신을 추가하거나 양자화로 최적화하면 된다. 학습 데이터가 너무 오래되었을 때는 재학습 후 서버를 버전 관리하면 된다. 모든 downstream 모델은 즉시 업데이트 결과를 얻을 수 있다. 마지막으로 모든 요청이 한 곳으로 옮으로써 GPU 서버는 적은 유휴 주기를 가져야하고 모든 비용이 잘 쓰여야 한다.

 통신 서버를 구축하기 위해 ZeroMQ와 그 파이썬 바인딩인 PyZMQ를 사용했다. 이건 가볍고 빠른 메세징 구현체이다. 양방향 메시지는 TCP/IPC/다르 프로토콜을 통해 보내고 받을 수 있다.
```python
import zmq
import zmq.decorators as zmqd

@zmqd.socket(zmq.PUSH)
def send(sock):
  sock.bind('tcp://*:5555')
  sock.send(b'hello')

# in another process
@zmqd.socket(zmq.PULL)
def recv(sock):
  sock.connect('tcp://localhost:5555')
  print(sock.recv())
```
- 빠른 추론 속도

구글이 공개한 BERT 코드는 학습과 평가를 제공한다. 그래서 다른 보조 노드들은 서빙전에 그래프에서 삭제되어야 한다. 또한 만약 k번째 레이어에서 풀링을 한다면, K+1 레이어부터 마지막 레이어까지의 파라미터를 추론에 필요가 없으니 삭제되어도 된다. 다음 그림은 일반적인 서빙전 과정이다.

<div aligh="center"> <img src="/image/bertasservice/5.png" /> </div>

구체적으로 freezing은 모든 tf.Variable을 tf.Constant로 바꾼다. Pruning은 불필요한 노드를 그래프에서 삭제하는 과정이다. 양자화는 모든 파라미터를 낮은 정확도의 자료형으로 바꾸는 것이다. tf.float32를 tf.float16이나 tf.uint8로 말이다. 현재 대부분의 양자화 방법은 모바일 장치를 위해 구현되어 있고 X86 장치에선 별다른 차이가 없을 수 있다.

텐서플로우는 위의 작업을 할 수 있는 API를 제공한다. 인풋과 아웃풋 노드만 지정해주면 된다.
```python
input_tensors = [input_ids, input_mask, input_type_ids]
output_tensors = [pooled]

from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.graph_util import convert_variables_to_constants

# get graph
tmp_g = tf.get_default_graph().as_graph_def()

sess = tf.Session()
# load parameters then freeze
sess.run(tf.global_variables_initializer())
tmp_g = convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors])

# pruning
dtypes = [n.dtype for n in input_tensors]
tmp_g = optimize_for_inference(tmp_g, [n.name[:-2] for n in input_tensors],
    [n.name[:-2] for n in output_tensors],
    [dtype.as_datatype_enum for dtype in dtypes], False)
    
with tf.gfile.GFile('optimized.graph', 'wb') as f:
    f.write(tmp_g.SerializeToString())
```
- 낮은 레이턴시로 서빙하기

우리는 매번 요청이 올 때마다 새 BERT 모델을 켤 필요가 없다. 대신, 처음에 한번 모델을 켜고 이벤트 루프에서 요청을 듣고 있으면 된다. `sess.run(feed_dict={...})` 가 한 방법이지만 충분하진 않다. 게다가 BERT 소스코드는 고수준 API인 tf.Estimator로 래핑되어서 이벤트 리스너를 넣을 방법이 필요하다. 가장 좋은 방법은 tf.Data의 제너레이터를 사용하는 것이다.
```python
def input_fn_builder(sock):
  def gen():
    while True:
      # recv req
      client_id, raw_msg = sock.recv_multipart()
      msg = jsonapi.loads(raw_msg)
      tmp_f = convert_lst_to_feauters(msg)
      yield {'client_id': client_id,
              'input_ids': [f.input_ids for f in tmp_f],
              'input_mask': [f.input_mask for f in tmp_f],
              'input_type_ids': [f.input_type_ids for f in tmp_f]}

  def input_fn():
          return (tf.data.Dataset.from_generator(gen,
              output_types={'input_ids': tf.int32, 'input_mask': tf.int32, 'input_type_ids': tf.int32, 'client_id': tf.string},
              output_shapes={'client_id': (), 'input_ids': (None, max_seq_len), 'input_mask': (None, max_seq_len),'input_type_ids': (None, max_seq_len)})
                  .prefetch(10))
  return input_fn

# initialize BERT model once
estimator = Estimator(model_fn=bert_model_fn)

# keep listen and predict
for result in estimator.predict(input_fn_builder(client), yield_single_example=False):
  send_back(result)
```
`estimator.predict()` 는 끝나는 않는 루프와 제너레이터를 리턴한다. 새로운 요청이 오면, 제너레이터를 데이터를 준비하고 estimator에 넘겨준다. 그렇지 않으면 제너레이터는 `sock.recv_multipart()`에 의해 다음 요청까지 블락된다.

`.prefetch(10)` 을 보면 prefetch 메커니즘을 추가하는건 효과적으로 배치 준비 시간을 없앨 수 있다. 모델이 추론을 하고 있느 도중 새 요청이 오면 prefetch를 안한다면 추론이 끝나기 전까지 아무 준비작업도 안하는 반면 `.prefetch(10)` 을 하면 추론중에도 10개의 배치를 준비할 것이다. 실제로 prefetch를 통해 10%의 속도 향상을 보았고 GPU 머신에서 효과적이다.

- 높은 확장성으로 서빙

다수의 클라이언트가 동시에 요청을 보낸다고 하자. 병행처리는 하나의 방법이지만 먼저, 서버가 어떻게 요청을 받아야하나? 먼저 서버는 첫 요청을 받고 결과가 올 때까지 연결하고 있어야한다. 그 다음에 두번째 요청을 수행한다면? 그러면 100명의 클라이언트가 있다면? 서버는 같은 방법으로 100개의 연결을 수행해야 할 것인가?

두번째 예로, 클라이언트가 10k개의 문장을 매 10ms마다 보낸다고 하자. 서버는 작업을 여러개로 나누어 병행처리하고 여러개의 GPU 워커를 할당한다. 그리고 다른 클라이언트는 한 문장을 1초마다 요청한다. 직관적으로, 이 작은 배치의 클라이언트는 결과를 바로 얻을 수 있어야 한다. 불행히도, 모든 GPU가 첫번쨰 클라이언트의 요청을 처리하느라 바쁘고 두번째 클라이언트는 절대 결과물을 얻지 못한다.

다수의 사용자가 한 서버에 연결할 때는 확장성과 로드밸런싱 이슈가 있다. bert-as-service에 나는 ventilator-worker-sink 파이프라인과 push/pull, pub/sub 소켓을 구현했다. ventilator는 배치 스케줄러와 로드 밸런서처럼 작동한다. 이 과정은 큰 요청을 작은 작업들로 나누고 워커에게 보내기 전에 이 작업들의 로드를 맞춘다. 워커는 작은 작업들을 ventilator로부터 받고 BERT 추론을 실행한 후 sink로 결과를 보낸다. sink는 모든 워커로부터 결과를 받고 ventilator로부터의 모든 요청의 완결성을 체크한 후 클라이언트에게 퍼블리싱한다. 

<div aligh="center"> <img src="/image/bertasservice/6.png" /> </div>

이 결과는 처음에 떠오른 아이디어는 아니고 몇번의 과정을 거쳐 나온 것이다. 위 과정을 좀 더 구체적으로 보면,

- 클라이언트로부터 보내고 받는 분리된 소켓. 보통의 REQ-REP 소켓과 비교해서 PUSH-PULL 소켓은 다음 PUSH 까지 REP을 기다리지 않는다. 클라이언트는 한번에 많은 요청을 보내고 받을 수 있다. 계산이 끝나면 서버는 결과를 PUB 소켓을 통해 보내고 클라이언트 ID를 헤더로 쓴다. 클라이언트는 SUB 소켓을 통해 듣고 매칭되는 구독을 받는다.
- PUSH-PULL과 PUB-SUB 디자인에는 적어도 두개의 이점이 있다. 먼저 서버와 클라이언트 사이의 연결을 계속 유지할 필요가 없다. 클라이언트로써 데이터를 지정된 장소에 놓으면 끝난다. 서버가 그걸 PULL을 통해 가져갈 것을 알기 떄문이다. 서버가 살아있는지 혹은 죽었는지 등을 신경 쓸 필요가 없다. 서버로써는, 지정된 장소에가서 작업을 받아온다. 누가 그 작업을 놓고 얼마나 많이 있는지는 상관없다. 최선을 다하고 모두 로드하면 된다. 받을 때도 마찬가지다, 모든 클라이언트는 지정된 장소에서 결과를 받는다. 이 디자인은 시스템을 더 확장가능하게 하고 강하게 만든다. bert-as-service에서 서버는 쉽게 켜지고 꺼질 수 있고 BertClient는 계속 작동할 수 있다.
- 두번쨰 이점은 비동기 인코딩과 멀티캐스팅 같은 멋진 특징을 사용할 수 있다는 것이다. 비동기 인코딩은 문장 전처리 시간과 인코딩 시간을 동시에 쓰고 싶을 때 유용하다. 이제 멀티캐스팅은 BertClient API에 구현되어서 같은 아이덴티티의 다수 클라이언트가 한 인코딩 결과를 동시에 다 받을 수 있다. 밑의 그림이 이 아이디어를 설명한다.

<div aligh="center"> <img src="/image/bertasservice/7.gif" /> </div>

- 당신들은 데이터가 파이프라인 아래로만 흐른다는 걸 알았을 것이다. 모든 메세지들은 위로 보내지지 않고 오직 아래의 다른 소켓으로만 보내진다. 그리고 받는이는 보낸이에게 메세지를 다시 보내지 않는다. 이렇게 back-chatter를 없애는 건 확장성에 꼭 필요하다. back-chatter를 없앨때, 모든 메세지의 흐름은 훨씬 간단해지고 non-blocking하게 되어 간단한 API와 프로토콜을 만들 수 있고 낮은 레이턴시를 갖게 만든다.
- ventilator와 워커 사이의 다수의 소켓이 있다. 요청 사이즈가 16 문장보다 작다면, ventilator는 작업을 첫 소켓으로 보낸다. 그렇지 않으면 ventilator는 작업을 작은 단위로 나눈 후 1~7 사이의 랜덤 소켓으로 보낸다. 워커는 8개의 소켓으로부터 계속 작업을 받고 낮은 소켓 번호부터 처리한다. 이것은 작은 요청이 크고 자주오는 요청에 의해 블락되지 않게 한다. 작은 요청은 항상 첫 소켓으로 보내지고 가장 먼저 처리된다. 다수의 클라이언트로부터 오는 크고 빈번한 요청은 서로 다른 소켓에 보내져 같은 확률로 워커에 할당되어 처리된다.
- ventilator, worker와 sink 의 프로세스가 분리되어 있다. 프로세스 레벨로 컴포넌트를 분리하는건 서버가 튼튼하고 레이턴시를 낮게 할 수 있게 한다. sink와 ventilator가 같은 모듈에 결합될 수 있지 않냐고 할 수 있지만, 결과 메세지는 프로세스를 지나 보내져야만 한다. 결국 통신 코스트를 줄이지 못한다. 게다가, 프로세스들을 분리하는건 전체 로직과 메세지 흐름을 간단하게 한다. 모든 컴포넌트는 하나의 작업에만 집중하고 쉽게 sink나 ventilator를 스케일링할 수 있다.

[깃헙](https://github.com/hanxiao/bert-as-service)에 벤치마크가 있으니 참고하길 바란다.

## 요약

bert-as-service가 2018년에 나오고 나서 커뮤니티로부터 꽤나 주목을 받았다. 1000개 이상의 스타를 한달만에 받기도 했다. 사람들은 나에게 감사하다고 메세지를 보내기도 한다. 개인적으로 딥러닝 모델을 프로덕션에 배포하는건 항상 좋은 학습 경험이다. 이 포스트에 쓰여진 것처럼, 이건 연구와 엔지니어링 두 사이드의 지식을 필요로 한다.