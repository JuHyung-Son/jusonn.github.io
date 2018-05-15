---
title: Differential calculus 리뷰
date: 2018-05-15
author: JuHyung Son
layout: post
tags:
  - Math
categories:
  - Deep_Learning
---

# Intro

사실 미분은 고등학생 때부터 쓰던 개념이라 정작 미분이 뭐냐고 물으면 잘 생각이 나지 않지만, 보통 미분을 말한다면 이렇게 얘기합니다. 함수 $f : \mathbb{R} \rightarrow \mathbb{R}$ 에서

$$f(x+h) - f(x) \approx f'(x)h$$

그리고 미분의 표기법은 상황이나 쓰는 방법에 따라 또 차이가 있습니다. 비슷하지만 모두 다른 것을 의미하는 표기죠. 보통 수업을 듣다보면 언제 어떤 표기를 쓰는지 경험적으로 알게 되는 거 같습니다.

$$f'(x), \frac{df}{dx}, \frac{ \partial f}{ \partial x}, \triangledown_{x} f$$

## Differential

### Def

$\mathbb{R^{n}}$ 에서 정의된 함수 $f : \mathbb{R^{n}} \rightarrow \mathbb{R}$ 를 봅시다. 이런 함수는 $x \in \mathbb{R^{n}}$ 과 아주 작은 $h$ 에서

$$f(x+h) = f(x) + d_{x} f(h) + o_{h \rightarrow 0} (h)$$

$$d_{x} f : \mathbb{R^{n}} \rightarrow \mathbb{R} : linear$$
$$\forall x, y \in \mathbb{R^{n}}, d_{x} f(x+y) = d_{x}f(x) + d_{x}f(y)$$

처럼 사용할 수 있을 때, **미분 가능** 하다고 합니다. 여기서 $d_{x} f$ 는 differential of f in x 라고 합니다.

## Gradients

모든 선형 형태의 $a: \mathbb{R^{n}} \rightarrow \mathbb{R}$ 에 대해 모든 $h \in \mathbb{R^{n}}$ 이고 다음을 만족하는 어떤 벡터 $u_{a} \in \mathbb{R^{n}}$ 가 존재합니다.

$$a(h) = <u_{a} | h> (< \cdot | \cdot > : scalar ~ product)$$

모든 선형 형태인 $a$ 는 벡터 $u_{a}, h$의 스칼라 곱으로 나타낼 수 있다는 말입니다.
즉, differential $d_{x} f$ 에서는 다음을 만족하는 벡터 $u \in \mathbb{R^{n}}$ 을 찾을 수 있습니다.

$$d_{x} (h) = <u|h>$$

### Def

이제 **Gradient** 를 정의합니다. gradient of f in x 는 다음과 같습니다.

$$ \triangledown_{x} f := u$$

위에 정의하였던 미분 식을 다시 쓰면,

$$f(x+h) = f(x) + < \triangledown_{x} f | h > + o_{h \rightarrow 0}(h)$$

### Example

예를 하나 봅시다. $f( \begin{pmatrix} x_{1} \\ x_{2} \end{pmatrix} ) = 3 x_{1} + x_{2}^{2}$ 인 함수 $f : \mathbb{R^{2}} \rightarrow \mathbb{R}$ 가 있다고 합니다. 다음,

$\begin{pmatrix} a \\ b \end{pmatrix} \in \mathbb{R^{2}}$ 와 $h = \begin{pmatrix} h_{1} \\ h_{2} \end{pmatrix} \in \mathbb{R^{2}}$ 를 하나씩 뽑아봅니다. 다음,

$$f( \begin{pmatrix} a+h_{1} \\ b+h_{2} \end{pmatrix} ) =3(a+h_{1}) + (b+h_{2})^{2}$$

$$= 3a + b^{2} 3h_{1} +2bh_{2} + h^{2}_{2}$$
$$= f(a,b) + 3 h_{1} + 2bh_{2} + o(h)$$
따라서 $d_{ \begin{pmatrix} a \\ b \end{pmatrix} } f ( \begin{pmatrix} h_{1} \\ h_{2} \end{pmatrix} ) = 3h_{1}+ 2bh_{2}$ 를 얻게 됩니다. 여기서 오른쪽의 식은 스칼라 곱으로 표현할 수 있죠.

$$d_{ \begin{pmatrix} a \\ b \end{pmatrix} } f ( \begin{pmatrix} h_{1} \\ h_{2} \end{pmatrix} ) = < \begin{pmatrix} 3 \\ 2b \end{pmatrix} \| \begin{pmatrix} h_{1} \\ h_{2} \end{pmatrix} >$$

따라서 여기서의 **gradient** 는 벡터 $u$ 가 되는

$$ \triangledown_{ \begin{pmatrix} a \\ b \end{pmatrix} } f = \begin{pmatrix} 3 \\ 2b \end{pmatrix}$$

## Partial Derivatives

변수 2개 이상일 때, 하나의 변수에 대해서만 미분하는 것이 편미분입니다. 편미분에서는 $\partial$ 기호를 사용합니다. 관습적으로 그런건지, 이유가 있는 모르겠네요.

$$\frac{\partial f}{\partial x_{i}} (x) = lim_{h \rightarrow 0} \frac{f(x_{1},...,x_{i} +h,...,x_{n}) - f(x_{1},...,x_{n})}{h}$$
$x \in \mathbb{R^{n}}$ 이지만 $\frac{\partial f}{\partial x_{i}}(x) \in \mathbb{R}$ 입니다. i번째 x에서의 변화량이죠.

편미분에서의 gradient는 위와 똑같지만 단지 요소가 더 많아질 뿐입니다.

$$\triangledown_{x} f = \sum^{n}_{i=1} \frac{\partial f}{\partial x_{i}} (x) e_{i}$$
$$= \begin{pmatrix} \frac{\partial f}{\partial x_{1}} (x) \\ \frac{\partial f}{\partial x_{2}} (x) \\ ... \\ \frac{\partial f}{\partial x_{n}} (x) \end{pmatrix}$$

$\frac{\partial f}{\partial x_{i}} (x)$ 는 f에서 i 번째 값의 편미분값을 의미합니다.

마지막으로 정리를 하면 이렇죠.

<div align="center"> <img src="image/differential/1.png" /> </div>

## Jacobian
