---
title: 추천 시스템 들어가기 (Collaborative Filtering, Singular Value Decomposition)
date: 2018-05-16
author: JuHyung Son
layout: post
tags:
  - Recommender
categories:
  - Studying
---

Steeve Huang의 Introduction to Recommender System. Part 1 (Collaborative Filtering, Singular Value Decomposition) 를 번역한 것입니다.

<a url="https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75"> 원문보기 </a>

# 1. Introductioin

추천 시스템은 물품들에 대한 유저의 선호도를 예측하고 유저에게 적합한 물품을 추천할 수 있는 시스템을 말합니다. 현대 사회에서 추천 시스템이 특히나 필요한 이유는, 인터넷의 발달로 사람들에게 주어진 선택지가 너무나 많기 때문입니다. 과거에는 실제 가게에서 쇼핑을 하였기 때문에 선택할 수 있는 가지 수가 제한되어 있었습니다. 예를 들어, 비디오 가게에서 선택할 수 있는 영화의 갯수는 가게에 사이즈에 달려있죠. 만면 요즘은 인터넷에서 어마어마하게 많은 양의 비디오를 제공하는 소스가 있습니다. 그 중 한가지인 넷플릭스는 또 비디오를 엄청나게 갖고 있죠. 접근가능한 정보의 양이 증가하면서 여기에 새로운 문제가 생겼습니다. 사람들이 자신이 원하는 것을 선택하는데에 어려움을 겪는다는 것입니다. 여기서 추천 시스템이 필요하게 되죠. 이번에는 추천 시스템의 간단한 두가지 방법을 소개합니다. Collaborative Filtering과 Singular Value Decomposition 입니다.

# 2. Traditional Approach

기존 추천 시스템에는 두가지 방법이 쓰였습니다.

- Content-based recommendation
- Collaborative Filtering

첫번째는 물품 자체를 분석하는 방법입니다. 예로, 유저에게 시를 추천할 때, 시 자체를 자연어처리 기법으로 분석해 추천하는 것이죠. 두번째 방법은 반대로 아이템이나 유저 자체에 대한 정보를 사용하지 않습니다. 단지 유저의 과거 행동 이력을 보고 추천하죠. 여기서는 collaborative filtering을 더 깊게 다뤄봅니다.

# 3. Collaborative Filtering

위에서 언급했듯이 Collaborative Filtering(CF)는 유저의 과거 행동에 기반한 추천 방법입니다. 여기에도 두가지 카테고리가 있죠.

- 유저 기반: 타겟 유저와 다른 유저간 비슷함의 정도를 이용합니다.
- 아이템 기반: 유저들이 아이템에 매겨놓은 점수, 다른 아이템과의 관계의 비슷함의 정도를 이용합니다.

CF의 기본적인 아이디어는 비슷한 유저들은 같은 흥미를 갖고 있고 비슷한 아이템을 좋아할 것이라는 겁니다.

m명의 유저와 n개의 아이템이 있다고 하면 $m*n$ 모양의 행렬을 이용해 유저들의 과거 행동을 나타냅니다. 행렬의 각 값들은 유저의 선호도가 들어갑니다. $M_{i,j}$ 는 유저 i의 아이템 j에 대한 선호도이죠. 이 행렬을 **utility matrix** 라고 부릅니다. CF는 유저가 아직 보지 않은 아이템들에 대한 utility matrix의 빈 값들을 채우는 것과 같죠. 유저의 선호도에는 **explicit opinion, implicit opinion** 두가지 종류가 있습니다. Explicit opinion은 한 유저의 한 아이템에 대한 선호도를 직접적으로 보여주는 것들입니다.(영화, 앱 레이팅) Implicit opinion은 반면 간접적으로 아이템에 대한 선호도를 나타내죠. (좋아요 갯수, 클릭 수, 방문 수) Explicit opinion는 값에 대한 해석이 필요하지 않으니 당연 더 직접적인 수치입니다. 예를 들어 한 유저가 어떤 곡을 매우 좋아합니다. 하지만 그 유저는 매우 바빠 딱 한번만 들었죠. 이럴 경우 Explicit opinion 없이는 유저의 그 곡에 대한 선호도를 추측하기 어렵습니다. 하지만 대부분의 사람들은 레이팅을 잘 주지 않죠. 우리가 가지는 데이터는 implicit opinion이 절대적으로 많습니다. 그래서 이런 간접적인 지표에 대해 적당한 피드백을 주는 것이 매우 중요합니다.

## User-based Collaborative Filtering

유저 기반 CF에서는 유저들 간의 비슷함 정도를 계산해야 한다고 했습니다. 그러면 이 비슷함을 어떻게 계산할까요? 두가지 방법이 있습니다. 피어슨 상관계수와 코사인 유사도입니다. $u_{i,k}$ 를 유저 i와 k의 유사도라고 하고 v_{i,j} 를 유저 i가 아이템 j에 준 레이팅이라고 합시다. 이 때 각 방법은 다음처럼 표현됩니다.

$$u_{ik} = \frac{\sum_{j} (v_{ij} - v_{i})(v_{kj} - v_{k})}{\sqrt{\sum_{j}(v_{ij} - v{i})^{2}} \sum_{j} (v_{kj} - v{k})^{2}}$$

<div align="center"> Pearson Correlation </div>

$$cos(u_{i}, u_{j}) = \frac{\sum^{m}_{k=1} v_{ik} v_{jk}}{\sqrt{\sum^{m}_{k=1} v^{2}_{ik} \sum^{m}_{k=1} v^{2}_{jk}}}$$

<div align="center"> Cosine Similarity </div>

이 두 방법은 모두 자주 쓰입니다. 차이점이라면 피어슨 계수는 모든 항목에 상수를 더해도 변화가 없다는 것이죠.

이제 레이팅 되지 않은 아이템에 대한 유저의 선호도를 아래의 식으로 예측해볼 수 있습니다.

$$v^{}_{ij} = K \sum_{v_{kj} \neq ?} u_{jk} v_{kj}$$

<div align="center">Unrated Item Prediction</div>

위 식을 예를 들면서 설명 해보겠습니다. 밑의 행렬에서 행은 유저를 나타내고 행은 영화를 나타냅니다. 마지막 행은 유저와 타겟 유저간의 유사도를 나타내고요. 각 값들은 유저가 영화에 매긴 레이팅입니다. 여기서는 유저 E를 타겟 유저라고 합시다.

|  | The Avengers | Sherlock | Transformers | Matrix | Titanic | Me Before You | Similarity(i, E) |
|:--:|:--:|:--:|:-:|:-:|:-:|:-:|:-:|
|  A  |  2  |   | 2  | 4  | 5 | | NA|
|  B  |  5  |   | 4  |   | | 1 | |
|  C  |    |   | 5  |   | 2| | |
| D  |   |  1 |   | 5  | | 4 | |
| E  |   |   | 4  |   |  | 2| 1|
| F  | 4  | 5  |   |  1 | | | NA|

유저 A와 F는 유저 E와 같은 영화에 대해 레이팅을 매긴 것이 없습니다. 그래서 A,F와 E 간에는 피어슨 계수가 정의되지 않습니다. 그러므로 B, C, D만을 보죠. 피어슨 계수를 이용해 다음과 같이 계산을 합니다.

|  | The Avengers | Sherlock | Transformers | Matrix | Titanic | Me Before You | Similarity(i, E) |
|:--:|:--:|:--:|:-:|:-:|:-:|:-:|:-:|
|  A  |  2  |   | 2  | 4  | 5 | | NA|
|  B  |  5  |   | 4  |   | | 1 | 0.87 |
|  C  |    |   | 5  |   | 2| | 1 |
| D  |   |  1 |   | 5  | | 4 | -1 |
| E  |   |   | 4  |   |  | 2| 1|
| F  | 4  | 5  |   |  1 | | | NA|

계수를 보니 D와 E는 매우 다른 유저임이 보이죠. D는 *Me Before You* 를 그의 레이팅 평균보다 높게 주었지만 E는 정반대입니다. 이제 위의 예측 식을 이용해 E가 레이팅을 하지 않은 영화들에 대해 예측을 해볼 수 있습니다.

|  | The Avengers | Sherlock | Transformers | Matrix | Titanic | Me Before You | Similarity(i, E) |
|:--:|:--:|:--:|:-:|:-:|:-:|:-:|:-:|
|  A  |  2  |   | 2  | 4  | 5 | | NA|
|  B  |  5  |   | 4  |   | | 1 | 0.87 |
|  C  |    |   | 5  |   | 2| | 1 |
| D  |   |  1 |   | 5  | | 4 | -1 |
| E  | 3.51*  | 3.81*  | 4  | 2.41*  | 2.48* | 2| 1|
| F  | 4  | 5  |   |  1 | | | NA|

유저 기반 CF는 이렇게 계산이 간단합니다. 하지만 맟가지 문제가 있기도 합니다. 가장 큰 문제는 유저의 선호도가 시간이 지나면서 변할 수 있다는 것이죠. 즉, 다른 유저들의 선호도에 기반해 한 유저의 선호도를 미리 측정하는 것은 안좋은 성능을 유발할 수 있습니다. 이런 상황을 완화하기 위해 아이템 기반 CF를 적용합니다.

## Item-based Collaborative Filtering

유저간의 유사도 대신 아이템 기반의 CF는 타겟 유저가 레이팅한 아이템들의 유사도를 기반으로 추천을 합니다. 여기서도 유사도는 피어슨계수와 코사인 유사도로 계산됩니다. 차이점이라면 아이템 기반 CF에서는 빈칸을 수직으로 채워넣습니다. 아이템 기반이니까요. 위와 같은 예를 들어 *Me Before You* 의 선호도를 예상해봅시다.

|  | The Avengers | Sherlock | Transformers | Matrix | Titanic | Me Before You |
|:--:|:--:|:--:|:-:|:-:|:-:|:-:|
|  A  |  2  |   | 2  | 4  | 5 | 2.94* |
|  B  |  5  |   | 4  |   | | 1 |
|  C  |    |   | 5  |   | 2| 2.48*|
| D  |   |  1 |   | 5  | | 4 |
| E  |   |   | 4  |   |  | 2|
| F  | 4  | 5  |   |  1 | | 1.12*|
|Similarity| -1  | -1  | 0.86  | 1  | 1  |   |

유저의 선호도는 유동적인데 반해 아이템은 덜 유동적이므로 유저 기반 CF에서 발생하는 문제를 해결했습니다. 그러나 여전히 몇가지 문제가 남아있습니다. 먼저, 가장 큰 문제는 확장성입니다. 유저와 아이템이 증가하면서 복잡도는 $O(mn)$ 으로 커집니다. 게다가 sparsity도 큰 문제입니다. 위의 표를 보면, *Matrix* 와 *Titanic* 모두를 레이팅한 유저는 한명밖에 없지만, 이 두 영화의 유사도는 1입니다. 좀 더 극단적으로 수십만의 유저가 있지만 두 유저만 매우 다른 영화에 비슷한 레이팅을 주어 매우 다른 영화임에도 불구하고 유사도가 비슷한 경우가 발생할 수 있죠.

# Singular Value Decomposition

위에서 발생한 두 문제를 다루는 방법으로는 **latent factor model** 을 사용해 유저와 아이템 간의 유사도를 찾는 방법입니다. 본질적으로는, 우리는 추천 문제를 최적화 문제로 바꾸고 싶은 것입니다. 한 유저에 대해서 어떤 아이템에 대한 레이팅을 얼마나 잘 예측하냐를 보는 것입니다. 이런 경우 가장 많이 쓰이는 방법이 Root Mean Square Error (RMSE) 죠. RMSE가 낮을수록 성능은 좋습니다. 아직 접하지 못한 아이템에 대해서는 레이팅을 모르니 아직은 그것들을 무시합니다. 일단은 utility matrix에서 아는 것에 대해서만 RMSE를 낮춰갑니다. 최소의 RMSE를 얻기 위해서 **Singular Value Decomposition** 이 쓰입니다.

$$\begin{pmatrix} x_{11} & x_{12} & ... & x_{1n} \\ x_{21} & ... \\ ... & ... \\ x_{m1} & ... & & x_{mn} \end{pmatrix} \approx \begin{pmatrix} u_{11} & ... & u_{1r} \\ ... & & \\ u_{m1} & ... & u_{mr} \end{pmatrix} \begin{pmatrix} s_{11} & 0 & ... \\ 0 & ... & \\ 0 & ... & s_{rr} \end{pmatrix} \begin{pmatrix} v_{11} & ... & v_{1n} \\ ... & & \\ v_{r1} & ... & v_{rn} \end{pmatrix}$$

$$\hat{X} \approx US V^{T}$$

<div align="center"> Singular Matrix </div>

X는 utility matrix, U는 left singular matrix 로 유저와 latent factor 간의 관계를 나타냅니다. S는 diagonal matrix로 각 latent factor의 중요함의 정도를 나타내고 $V^{T}$ 는 right singular matrix로 아이템과 latent matrix 간의 유사도를 나타냅니다. 계속 언급되는 **latent vector** 가 뭘까요? 이건 유저나 아이템이 갖는 특성과 같은 것을 좀 더 넓은 범위로 설명하는 겁니다. 예를 들명, 음악에서 latent factor 는 음악이 속한 장르와 같은 것들을 포함하죠. SVD는 utility matrix에서 latent factor를 추출하면서 차원을 감소시킵니다. 본질적으로, 각각의 유저와 아이템을 r차원을 가진 latent space로 옮기는 거죠. 그렇게 하므로 유저와 아이템이 직접 비교가 가능해지면서 그들간의 관계를 더 이해하기 쉽게 됩니다. 밑의 그림을 보면 이해가 쉽죠.

<div align="center"> <img src="/image/recommender/1.png"/> </div>

SVD는 최소의 SSE를 얻는다는 굉장히 좋은 특성이 있습니다. 그래서 차원 축소에도 자주 쓰이는 방법입니다. 아래의 식에서 A를 X로 $\Sigma$ 를 S로 바꾼 것입니다.

$$min_{U,V,S} \sum_{ij \in X} (X_{ij} - [USV^{T}]_ {ij})^{2}$$

그런데 이것이 RMSE와 어떻게 연관이 있을까요? RMSE와 SSE는 monotonic한 연관이 있다고 밝혀졌습니다. 즉 SSE가 낮으면 RMSE도 낮은 것이죠. SVD가 SSE를 최소로 한다는 특성 덕분에, RMSE 또한 최소가 됩니다. 그렇기 때문에 SVD는 아주 좋은 최적화 툴이죠. 이제 유저가 알지 못하는 아이템에 대해 예측을 하기 위해선 $U, \Sigma , T$를 곱합니다.

파이썬의 Scipy를 사용해 간단히 svd를 사용할 수 있습니다.
```python
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

A = csc_matrix([[1,0,0],[5,0,2],[0,-1,0],[0,0,3]], dtype = float)
u,s,vt = svds(A, k=2)
s
>> array([2.75193379, 5.6069665])
```

SVD는 CF에서 확장성과 sparsity 문제를 해결합니다. 그러나 SVD도 단점이 없는 건 아니죠. SVD의 주요 문제점은 우리가 유저에게 아이템을 추천하는데 어떠한 설명이나 이유가 없다는 것입니다. 만일 유저가 왜 그 상품을 추천 받았는지 알고 싶어한다면 이것은 아주 큰 문제가 됩니다. 이것에 대해선 다음 포스트에서 다루죠.

# 5. Conclusion

추천 시스템에서 사용되는 Collaborative Filtering, Singular Value Decomposition 두가지 방법을 다뤘습니다. 다음 포스트에서는 더 진화된 알고리즘을 다루겠습니다.
