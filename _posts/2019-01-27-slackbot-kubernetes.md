---
title: 쿠버네티스에 올리는 nodejs 슬랙봇
date: 2019-01-23
author: JuHyung Son
layout: post
tags:
  - kubernetes
---

구글 스터디잼 쿠버네티스 입문 과정을 수강 중
구글 클라우드를 통해 자습형 실습을 해보았다.

사실 저번부턴 뭐가 진행되는지 이해 못하고 따라 치기만 하고 있다.

이번에 학습한 내용은
1. nodejs 로 만든 슬랙봇을 올리기
2. 슬랙봇 도커라이즈, 쿠버네티스에 올리기

먼저 자신의 사용하는 슬랙이나 팀을 하나 만들어 슬랙 앱을 만들어야 한다.
봇도 만들어서 bot OAuth 키를 복사해놓자
`https://api.slack.com/apps`

샘플 코드는 아래 레포에 있다.

`git clone https://github.com/googlecodelabs/cloud-slack-bot.git`

샘플 코드 kittenbot.js 안에 token을 붙여넣고 실행해보면 앱이 실행된다. 슬랙에서 kitten-bot과 대화가 가능해진다.
끄면 앱도 꺼짐.

이 앱을 컨테이너화 해보자. 먼저 Dockerfile을 작성

```bash
FROM node:5.4
COPY package.json /src/package.json
WORKDIR /src
RUN npm install
COPY kittenbot.js /src
CMD ["node", "/src/kittenbot.js"]
```

그리고 `docker build` 로 이미지를 빌드해준다. 이미지가 잘 빌드되었다면 로컬에서 `docker run` 을 통해 실행해 볼 수 있다.
이 과정은 구글 스터디잼 과정이라 구글 컨테이너 레지스트리에 푸시를 해본다.

`gcloud docker -- push gcr.io/${PROJECT_ID}/slack-codelab:v1`

꽤 오래걸린다.

이제 도커 이미지가 온라인상에 있고 쿠버네티스를 통해 관리할 수 있다. 쿠버네티스 클러스터를 만들고 배포도 만들어 여러개의 앱을 관리해보자.

```bash
# 클러스터
gcloud container clusters create my-cluster \
      --num-nodes=2 \
      --zone=us-central1-f \
      --machine-type n1-standard-1

# 배포
kubectl create secret generic slack-token --from-file=./slack-token
```

