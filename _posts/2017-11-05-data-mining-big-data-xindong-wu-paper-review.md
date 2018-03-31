---
id: 165
title: 'Data Mining with Big Data, Xindong Wu &#8211; Paper review'
date: 2017-11-05T21:07:10+00:00
author: JuHyung Son
layout: post
permalink: /data-mining-big-data-xindong-wu-paper-review/
dsq_thread_id:
  - "6264113280"
image: /wp-content/uploads/2017/11/R1920x0-250x250.png
categories:
  - Paper
---
<h1>Data Mining with Big Data</h1>
빅 데이터(Big Data)는 큰 사이즈,복잡도, 어떠한 복잡한 소스로 부터 나오는 것들을 고려한다. 네트워크, 저장 기술의 발달로 빅데이터는 과학, 공학 분야에서 빠르게 성장하고 있다. 이 논문에서는 HACE theorem이라는 빅데이터가 가지는 특징을 특성화한 정리를 소개한다.

학자들은 빅데이터의 시대가 왔다고 한다. 매일 2.5 quintillion bytes의 데이터가 생성되고 오늘 생성된 90%의 데이터는 지난 2년간 생성된 데이터와 맞먹는다. (IBM 2012) 2012년 9월 4일, 미국 전대통령 오바마와 롬니의 논쟁 후 2시간만에 천만개의 트윗이 트윗되었다. 또 사진 공유 사이트인 Flickr에서는 하루에 18만개의 사진이 업로드된다. 사진 하나의 사이즈가 2MB라고 한다면, 하루에 3.6테라바이트의 정보가 저장되는 것이다. 데이터는 점점 더 많이 저장되고 적용된다. 빅데이터 적용에서 가장 큰 이슈는 큰 사이즈의 데이터를 다루고 유용한 정보만을 추출하는 방법이다. 저장되는 대부분의 데이터는 쓸모없는 것이기 때문에 가치있는 정보를 추출하는 것은 매우 효율적이고 실시간이어야 한다. 한 예로 Square Kilometer Array(SKA)는 1초에 40기가의 데이터를 생성한다. 이런 엄청난 데이터 양 때문에 연구자들은 특정한 패턴 찾는 연구를 한다. 이러한 전에 없는 데이터 덕분에 지금의 시대는 효과적인 데이터 분석과 예측 플랫폼을 요구한다.
<h2>HACE Theorem</h2>
빅데이터는 큰 사이즈, 전에 없는 새로운 형태, 특정한 출처, 분산 제어로부터 시작하고 복잡하고 진화가 가능하다. (Big Data starts with large-volume, <b>h</b>eterogeneous, <b>a</b>utonomous sources with distributed and decentralized control, and seeks to explore <b>c</b>omplex and <b>e</b>volving relationships among data.)

<img class="aligncenter wp-image-166 size-full" src="http://dllab.xyz/wp-content/uploads/2017/11/R1920x0.png" alt="" width="547" height="311" />

저자는 코끼리 이야기를 예로 든다. 여러명의 장님이 아주 큰 코끼리를 판단하려고 한다고 하자.  이 아주 큰 코끼리가 빅데이터이다.  장님의 목표는 그가 수집한 정보를 이용해 코끼리를 그리는 것이다. 각자의 정보가 매우 제한적이기 때문에 각각의 장님들이 서로 다른 것들을 그려도 이상하지 않다. 이제 코끼리가 매우 빠르게 성장하고 자세도 계속 변하며 이제 장님들은 서로의 정보를 공유한다고 하자. 이제 문제는 좀 더 현실의 빅데이터 문제와 비슷해졌다. 장님들은 자신의 정보를 공유하고 협력함으로 좀 더 코끼리와 가까운 그림을 그릴 것이다. 하지만 현실의 문제는 확실히 장님들이 코끼리 정보를 공유하고 전문가를 불러 코끼리를 그리는 것처럼 간단하지는 않다. 이제 저자는 HACE에 나온 빅데이터의 특징을 설명하는데 어떠한 것들이 있는지만 보자.

- Huge Data with heterogeneous and Diverse Dimensionality

- Autonomous Sources with Distributed and Decentralized Control

- Complex and Evolving Relationships
<h2>Data Mining Challenges with Big Data</h2>
이 장에서 저자는 데이터 마이닝 Challenge를 설명하는데 3가지 티어로 설명한다. 빅데이터 문제를 해결해 나가는 순서이다.

- Tier 1: Big data Mining Platform.

데이터가 커짐에 따라 개인 컴퓨터에서 가능하던 작업들이 불가능하게 된다. 이제 따라 여러가지 빅데이터 플랫폼이 등장하는데 데이터를 분산처리하는 Mapreduce, 슈퍼 컴퓨터를 소개한다.

- Tier 2: Data privacy and domain knowledge.

정보 공개, 개인 정보 문제, 도메인 지식 등을 소개한다.

- Tier 3: Big Data mining algorithms.

데이터 분석, 마이닝을 하는 것들을 소개한다.

빅데이터를 다루는 것은 아직도 꽤나 도전적인 일이다. 빅데이터라는 단어 자체는 데이터의 사이즈를 말하는 것이지만 HACE 정리는 빅데이터의 특징을 제시한다. Heterogeneous, Autonomous, Complex, Evolving 이 그것이다. 또한 빅데이터는 데이터 레벨의 전문가, 시스템 레벨의 전문가, 모델 레벨의 전문가가 협업해야하는 분야이다. 이젠 일반인들도 사회, 경제적 이익을 위해 데이터 생산에 참여하는 단계까지 이르렀다. 빅데이터의 시대가 온 것이다.

<a href="http://ieeexplore.ieee.org/document/6547630/">논문 원문 보기</a>
