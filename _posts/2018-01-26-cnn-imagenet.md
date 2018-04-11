---
id: 605
title: 'CNN- Imagenet에 쓰인 주요 모델'
date: 2018-01-26T19:14:05+00:00
author: JuHyung Son
layout: post
categories:
  - Deep_Learning
tags:
  - Vision
  - cnn
---
<h2>ILSVRC</h2>

예전에 ILSVRC라는 유명했던 이미지 분류 대회가 이미지 분야에서 유행했습니다. 현재 이 대회는 공식적으로 종료되었고 캐글에서 대회를 이어가고 있습니다. 대회는 Imagenet이라는 데이터를 사용하는데 1000개의 카테고리와 수백만의 이미지 데이터로 이루어져 있습니다. ILSVRC는 단지 이미지 분류만이 있는 게 아니고 object detection 등 몇가지가 더 있기도 하죠. 대회가 종료된 이유는 2012년 딥러닝을 이용한 모델이 우승을 한 것이 시발점이 되었습니다. 이후 거의 모든 팀이 딥러닝 모델을 들고 나왔고 인간보다 정확한 결과가 나오고 나서부터는 업그레이드 버전이 아닌 옆그레이드 버전이 무수히 쏟아졌기 때문이라고 합니다.

<div align='center'><img src="/wp-content/uploads/2018/01/스크린샷-2018-01-26-오후-6.44.32.png" alt="" width="2798" height="1400" /> </div>

<h2>CNNs</h2>

ILSVRC에서 우승을 했거나 유명한 CNN들은 ResNet, VGG16, VGG19, Inception_V3, Xception 이 있습니다. 그리고 이 모델들은 케라스에서 학습된 결과를 제공하는 모델들입니다. 아마 텐서플로우에도 있겠지만 케라스는 위의 모델들을 불러와서 분류를 하거나 transfer learning을 할 수 있는 모델이 있습니다. GPU가 없는 저로써는 아주 감사한 일이죠. 이것이 가능한 이유는 딥러닝은 paramerized learning 이기 때문입니다. 데이터를 저장하거나 뭔가 다른 것이 필요한 게 아닌 파라미터 값만을 저장하면 되기 때문입니다. 쉽게 생각하면, 학습된 어떤 $f(x)$를 저장하는 겁니다.

<h3>VGG16, 19</h3>

VGG16,19는 2014년 CNN이 태동하던 시기에 나온 초기 모델입니다. 이름에서 보듯이 각각 16, 19개의 레이어를 가지고 있습니다. 2014년 당시에는 엄청나게 Deep한 모델이었다고 하죠. 초기 모델이라 어떤 새로운 것이 있기 보다는 단지 몸집을 엄청나게 불린 모델입니다. 모델은 Conv, Pooling, FC layer만으로 이뤄져 있기 때문에 직접 코딩하기 쉬운 모델입니다. VGG는 16개의 레이어 밖에 없기 때문에 1000개 레이어까지 등장한 지금보기엔 매우 가벼운 모델로 보이지만 사실 직접해보면 그렇지 않습니다.

VGG는 꽤 큰 단점이 있는데, 먼저 학습이 매우 느리다는 겁니다. 그 이유는 파라미터가 많기 때문입니다. 두번째 단점도 많은 파라미터 때문에 생기는 단점으로, 용량이 엄청나게 크다는 것이다. VGG는 각각 533MB, 574MB 이다. 케라스로 직접 받으면 상당히 오래 걸리고 하드 용량이 1기가가 줄어듭니다... 다행히 다음 모델들부터는 이 단점이 개선됩니다.

<div align='center'> <img src="/wp-content/uploads/2018/01/스크린샷-2018-01-26-오후-6.44.52.png" alt="" width="972" height="1546" /> </div>

<h3>ResNet</h3>

He et al. 이 발표한 ResNet. He initialization의 He와 같은 사람인지는 모르겠습니다. 옆그레이드가 아닌 업그레이드가 이뤄졌습니다. ResNet은 레이어 자체는 굉장히 많지만, 사실 모델의 용량은 ResNet50 기준으로 102MB 입니다. 모델에 대한 자세한 내용은 다음에 하겠습니다.

<div align='center'><img src="/wp-content/uploads/2018/01/스크린샷-2018-01-26-오후-6.45.49.png" alt="" width="1780" height="1638" /> </div>

<h3>Inception V3</h3>

CNN의 아버지 Yann LeCun이 구글에서 만든 모델입니다. GoogLeNet이라고도 불리고 Inception V3라고도 불립니다. 여기서는 Inception 모듈이라는 것으로 화제가 되었죠. Inception 모듈은 "multi level feature extracting"을 하는 모듈입니다. 이 모델은 93MB의 크기를 가집니다. 이후에 이 모델을 확장한 Xception이 나오는데 더 좋은 성능과 91MB라는 약간 작은 용량을 가집니다.
<div align='center'> <img src="/wp-content/uploads/2018/01/스크린샷-2018-01-26-오후-6.45.09.png" alt="" width="1188" height="732" /> <img src="/wp-content/uploads/2018/01/스크린샷-2018-01-26-오후-6.45.38.png" alt="" width="3012" height="1056" /></div>

<h2>Comparation</h2>
이외에도 여러 Net들이 존재하고 현재도 아마 나오고 있습니다. 이 CNN들은 좋은 성능과 작은 용량을 가지는 쪽으로 발전 중입니다. 예를 들면, SqueezeNet이라는 모델은 4.9MB의 크기 밖에 안되는 작은 모델로 칩에 내장되어 쓰이는 것을 목표로 개발되었습니다. 아래는 모델의 성능과 용량을 보여주는 표입니다.

<div align='center'><img src="/wp-content/uploads/2018/01/스크린샷-2018-01-26-오후-6.46.03.png" alt="" width="1514" height="1164" /> </div>

다음엔 이 CNN들은 하나씩 코드로 짜보려고 한다.

참조

CS231n

DL4CV
