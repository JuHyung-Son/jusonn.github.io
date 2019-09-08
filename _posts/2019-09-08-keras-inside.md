---
title: 인사이드 tf 케라스
date: 2019-09-08
author: JuHyung Son
layout: post
tags:
  - ml
---

구글 내부에서 진행하는 tf keras 트레이닝을 보고 간단하게 정리해봤다.
[유투브](https://www.youtube.com/watch?v=UYRBHFAvLSs&list=PLQY2H8rRoyvzIuB8rZXs7pfyjiSUs8Vza&index=7)
그 동안 tf 2.0 의 케라스 api를 쓰면서 헷갈렸던 점, 궁금했던 점을 위주로 본다.
TF 2.0 에서 케라스는 공식 high level api 로 들어있고 공식 문서에서도 tf.keras를 통해 모델을 짜는 것을 추천하고 튜토리얼마저 keras api를 사용한다.
물론 케라스 api를 쓰지 않고 tf low level api를 사용할 수는 있지만, 굳이 그래야할 이유나 상황이 있을진 모르겠다.

케라스는 크게 이렇게 구성되어 있다.
1. Layer class
2. Model class
3. functional API
4. loss, metric, callback 등등

가장 먼저 Layer class는 네트워크에서의 레이어의 추상화된 클래스이다. 케라스의 거의 모든 것은 이 레이어에 관한 것이거나 이것에 상호작용하는 것들이다.

## tf.keras는 뭘 하는 앤가?
1. 배치 계산
  1. 그래프 모드와 eager 모드 둘 다 가능 (유저가 작성한 레이어는 eage만 가능)
  2. 학습, 추론 모드 (dropout, batchnorm 에서 작동이 다르다.)
  3. 마스킹 (time serires)
2. 상태관리 (trainable weights, non-trainable weights)
3. loss, metric 추적
4. 타입 체크 (shape)
5. frozen, unfrozen 가능
6. serialized, unserialized 가능
7. DAG 작성

## 그럼 tf.keras 뭘 못하나?
1. gradients
2. device placements
3. 분산 학습
4. N개 샘플의 텐서로 시작하지 않는 것.
5. 타입 체크 (데이터셋, batch 계산이 아닌것, 아웃풋 혹은 인풋이 없는 동작)

이제 케라스를 이용해 케라스의 Dense 역할을 하는 linear layer를 두가지 방식으로 짜보자.

<div aligh="center"> <img src="/image/keras/1.png" /> </div>

가장 간단한 방식의 코드이다. `linear_layer = Linear(4, input_dim=2)` 를 실행하면 __init__이 실행되면서 필요한 파라미터들이 만들어지고 `y=linear_layer(x)` 에서 call 메소드가 실행되면서 포워드 계산이 이루어진다. 파이토치를 써봤다면 상당히 익숙하고 거의 똑같다.
위 코드의 불편한 점은 파이토치처럼 input_dim을 다 넣어주어야 한다는 점이다. 모델이 커지면 이 과정이 얼마나 귀찮은지는 다들 알 것이다.
그럼 좀 더 케라스답게 위 코드를 다시 써보자.

<div aligh="center"> <img src="/image/keras/2.png" /> </div>

위 코드에선 `linear_layer = Linear(32)` input_dim을 넣을 필요가 없다. 위 코드를 실행하면 어떤 파라미터가 생기지도 계산되지도 않는다. 다음 위 레이어를 실행하면 `y = linear_layer(x)` call 메소드가 실행되기에 앞서 build 메소드가 내부적으로 먼저 실행된다. build 메소드는 말그대로 레이어에 필요한 것들은 빌드한다. 위에선 두 파라미터들을 만들었는데 내부에서 input_shape이 자동적으로 인자로 들어가게 된다. input_dim을 일일히 계산하면 넣을 필요가 없어지는 방식이다.

위 두가지 방식의 가장 큰 차이는 파라미터를 __init__ 에서 만드느냐 아니냐의 차이이다. 케라스 문서는 학습해야할 파라미터를 생성하는 것들은 build 메소드 안에서 하길 권장한다. 학습에 필요하지 않고 저장되어야할 파라미터는 다음과 같이 작성하게 된다.

<div aligh="center"> <img src="/image/keras/3.png" /> </div>

그럼 케라스에 내장된 레이어를 사용해 더 큰 레이어를 만드는 경우에는 어떻게 될까? 
3개의 linear 레이어로 구성된 MLP 레이어를 보자.

<div aligh="center"> <img src="/image/keras/4.png" /> </div>

위 코드의 `Linear`는 위에서 만든 두번째 클래스 레이어이다. 위 코드에선 __init__ 에서 레이어를 정의하였다. 하지만 이 과정에서 어떠한 파라미터도 만들어지지 않는데 그 이유는 `Linear()` 안에 build 메소드가 실행되지 않았기 때문이다. `y = mlp(tf.ones(shape=(3, 64))`를 실행하면 `MLPBlock()` 의 call 메소드가 실행되고 그 첫줄인 `x = self.linear_1(inputs)` 에서 `Linear()` 의 build 메소드가 실행되고 call 메소드가 실행되게 된다.

## 학습
위에서 만든 `Linear` 를 가지고 학습을 해보자.
학습은 직접 루프를 짜는 방법과 fit을 사용한 방법이 있다. 일단 직접 짜보자.

<div aligh="center"> <img src="/image/keras/5.png" /> </div>

tf 2.0 부터는 세션이 없어지고 위처럼 루프를 짜게 되며 `tf.GradientTape()` 이라는 걸로 그래디언트를 추적한다. 상당히 파이토치스러워 진 모습니다. @tf.function 을 사용해 그래프모드로 실행할 수도 있다. 만약 네트워크 중간의 결과물로 loss를 만든다면 아래처럼도 가능한다.

## 학습 중 모델 내부에서 loss 계산

<div aligh="center"> <img src="/image/keras/6.png" /> </div>

`self.add_loss` 를 사용하면 된다. 이 메소드를 실행하면 결과값이 losses 리스트에 추가되며 이 리스트는 매 call 메소드 호출마다 리셋되므로 마자막 학습 루프에서처럼 쓰인다.

## 학습, 추론 모드
학습에서와 추론에서 동작이 다르다면? dropout, batchnormalization이 그 예인데 이는 call 인자 중 training 인자를 사용해 해결한다. batchnormalization 코드를 보면 쉽게 이해 가능하다.

<div aligh="center"> <img src="/image/keras/7.png" /> </div>

# Model

지금까지는 레이어를 짯다. 위 코드를 보면 모두 `Layer`를 상속 받았음을 볼 수 있다. 이번엔 `Model`을 이용해 MLP 모델을 짜보자.

<div aligh="center"> <img src="/image/keras/8.png" /> </div>

언뜻보기에 똑같다. 사실 보기에 똑같은게 아니고 정말 똑같다. `Model` 클래스는 `Layer` 클래스를 상속받았기 때문이다. 
`Model` 클래스는 단지 `Layer` 클래스 + a 이다. 모델 클래스에 추가된 것들은, 
1. 학습 (.compile, .fit, .evaluate, .predict)
2. 저장 (.save)
3. 서머리, 플롯 (.summary, .plot_model)

강의에 따르면 `Layer` 클래스는 말그대로 레이어, 즉 MLP, resnet, inception block 과 같은 것들은 만들때 사용하고 `Model` 은 말 그대로 전체 모델을 만들때 쓰는 것이다.

여기서 compile(), fit()은 디폴트로 그래프 모드로 실행되고 eager도 가능하다.

# .compile()
compile 메소드로 모델을 컴파일 했을땐 다음과 같은 과정을 거치게 된다.
1. compile 인자 체크
2. losses와 output 매필
3. metric과 output 매핑, 모델 endpoint 정의
4. 전체 loss 준비
5. trainable weight, gradient 얻기
6. 인풋, 아웃풋 준비

#.fit()
다음 fit 메소드를 실행하면 학습이 다음처럼 일어나게 된다.
1. 데이터 체크
2. 콜백 셋업
3. `on_train_begin` 콜백
4. `on_epoch_begin` 콜백
5. `on_batch_begin` 콜백
6. 함수 실행, 아웃풋, 로스 계산
7. `on_batch_end` 콜백
8. 검증 실행
9. `on_epoch_end` 콜백
10. 메트릭 리셋
11. `on_train_end` 콜백


# Functional API
나는 평소에 케라스 functional api를 써보지 않았고, 이것으로 구현된 코드도 잘 못봤다. 하지만 functional api를 잘쓰면 매우 좋을 거 같다. functional api는 Directed acyclic graph (DAG) 를 만드는 방법이다. DAG는 우리가 보통 보는 모델 네트워크로 보면 된다.
<div aligh="center"> <img src="/image/keras/dag.png" /> </div>

<div aligh="center"> <img src="/image/keras/9.png" /> </div>

functional api 로는 이런 식으로 모델을 짤 수 있다. 간단하게 그래프가 시작될 인풋과 아웃풋을 중심으로 짜고 그 사이를 여러 레이어로 연결하는 방식이다. 물론 이 과정에서 build, call 메소드는 알아서 실행된다. 그리고 여기서의 인풋은 단지 shape만을 정의하고 실제 데이터를 넣지는 않늗다. functional api의 특징들은 다음과 같다.
1. dag 레이어를 연결하는 api
2. 사용하기 쉬움, 개발자보다 더 넓은 층의 사용자를 염두해두고 만든 것.
3. 선언적임
4. 디버깅은 construction 과정에서 함. (사실 파이썬을 쓰는게 아니고 선언만 하는 것, 에러가 있다면 DAG를 잘못 만들었을 떄의 에러임)

작성중..