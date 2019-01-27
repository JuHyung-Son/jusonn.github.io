---
title: 간단히 보는 Gaussian Process 
date: 2019-01-23
author: JuHyung Son
layout: post
tags:
  - ml
categories:
  - ml
---

보통 기계학습 알고리즘은 어떤 알려지지 않은 분포로부터 뽑힌 샘플 (학습 데이터셋)을 가지고 1. convex 최적화 문제를 풀어 데이터에 가장 적합한 모델을 만들고 이 모델로 테스트 데이터에 대해서 최고의 추론을 하는 것들을 말한다.

반면 베이지안 방법은 데이터에 가장 적합한 모델을 찾는 것이 아니라 모델의 사후 분포를 계산한다. 이 사후 분포는 모델 결과값의 불확실성을 정량화하는 방법을 제공해 주고 이 불확실성에 대한 정보로 새로운 테스트 셋에 대해 더 견고한 모델을 만들게 해준다.

이해가 쉬운 회귀분석은 인풋 공간 $X = \mathbb{R}^n$ 에서 아웃풋 공간 $Y = \mathbb{R}$ 로의 매핑을 학습하는 문제다. 여기서는 특히 커널 기반의 베이지안인 가우시안 프로세스를 보자. 

# 다변수 가우시안 

확률변수 $x \in \mathbb{R}^n$ 은 다음을 만족하면 평균이 $\mu \in \mathbb{R}^n$ 이고 공분산이 $\Sigma \in \mathbb{S}^{n}_{++}$ 인 다변수 정규분포이다.

$$p(x; \mu, \Sigma) = \frac{1}{(2 \pi)^{n/2} |\Sigma|^{1/2}} exp (-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu))$$

표기는 $x \sim N(\mu, \Sigma)$ 와 같이 한다.

## 다변수 가우시안 properties

다변수 가우시안의 다음의 property를 따른다.

1. Normalization
pdf의 정규화: $\int_{x} p(x; \mu, \Sigma) dx = 1#

2. Mariginalization
- p(x_{A} = \int_{x_{B} p(x_{A}, x_{B}; \mu, \Sigma)} dx_{B})
- p(x_{B} = \iny(x_{A} p(x_{A}, x_{B}; \mu, \Sigma)) dx_{A})
는 가우시안이다.

3. Conditioning
conditional densities: 
$$p(x_{A} | x_{B}) = \frac{p(x_{A}, x_{B}; \mu, \Sigma)}{\int_{x_{A}} p(x_{A}, x_{B}; \mu, \Sigma)} d x_{A}$$
$$p(x_{B} | x_{A}) = \frac{p(x_{A}, x_{B}; \mu, \Sigma)}{\int_{x_{B}} p(x_{A}, x_{B}; \mu, \Sigma)} d x_{B}$$
도 가우시안이다.

4. Summation 가우시안 변수의 합 역시 가우시안이다.
$y \sim N(\mu, \Simga)$, $\z \sim N(\mu` , \Simga`)$ 의 합 $y+z \sim N(\mu + \mu` , \Sigma + \Sigma`$ 은 가우시안이다.

# 가우시안 프로세스

가우시안 프로세스는 다변수 가우시안의 랜덤 함수의 분포에 대한 무한대로의 확장판이다. 
만자 힘수에 대한 확률 분포를 어떻게 만드는 지보자.

유한한 집합 $X = \{ x_{1}, ..., x_{m} \}$ 과 $X$ 를 $\mathbb{R}$ 로 보내는 매핑의 집합 $H$ 가 있다고 하자.