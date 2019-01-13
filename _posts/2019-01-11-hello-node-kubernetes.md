---
title: 쿠버네티스 노드 소개
date: 2019-01-11
author: JuHyung Son
layout: post
tags:
  - tech
---

### 구글 스터디 잼 입문반: Kubernetes in the google cloud를 수강 중입니다.


쿠버네티스에 대한 소개는 여기를 봐봅시다. 
https://kubernetes.io/

쿠버네티스는 자동 배포, 스케일링, 컨테이터 어플리케이션 관리를 하는 오픈소스 시스템이라고 하네요.

여기서는 코드를 쿠버네티스에서 실행되는 복제 어플리케이션으로 변환해봅니다.

- 노드 서버를 만들고 
- 도커 컨테이너 이미지를 만들고
- 컨테이너 클러스터를 만들고 
- 쿠버네티스 포드를 만들고 
- 서비스의 규모를 확장해 볼 겁니다.

저는 사실 뭘 한다는건지 잘 모르겠는데요. 일단 따라해 보겠습니다.

예제에서 서비스할 어플리케이션은 hello-world 를 띄우는 서버입니다.
일단 서버를 만듭시다.

```javascript
var http = require('http');
var handleRequest = function(request, response) {
  response.writeHead(200);
  response.end("Hello World!");
}
var www = http.createServer(handleRequest);
www.listen(8080);
```

이제 ```node server.js``` 를 실행하면 로컬에서 hello world! 를 보여주겠죠.

다음 이 서버 코드를 실행하는 도커 컨테이너 이미지를 만듭니다.

Dockerfile 을 다음처럼 만들고 빌드하고 실행하면 완성!

```
FROM node:6.9.2
EXPOSE 8080
COPY server.js .
CMD node server.js
```

```bash
# hello-node로 빌드 
docker build -t gcr.io/project_id/hello-node:v1 .

# 컨테이너 이미지 실행 
docker run -d -p 8080:8080 gcr.io/project_id/hello-node:v1
```

도커 이미지를 실행했으니 로컬호스트:8080에 들어가면 방금과 같은 페이지가 보일겁니다. 이미지가 잘 작동한다면요.

이미지를 구글 컨테이너 레지스트리에 올려봅니다.
```bash
gcloud docker -- push gcr.io/project_id/hello-node:v1
```

오 처음으로 어딘가에 이미지를 푸쉬해봤네요. 이제는 컨테이너 클러스터를 만들 차례입니다.
저는 공학적 지식이 부족해 여기서부터는 뭘 한다는건지 잘 모르겠어요. 설명에 따
르면,
하나의 클러스터는 구글에서 호스팅하는 쿠버테니스 마스터 api 서버와 일련의 작업자 노드로 구성된데요. 그리고 작업자 노드는 compute engine 가상 머신입니다. 컨테이너들은 가상 머신생성과 유지를 자동화하는 클러스터에서 관리됩니다. 라고 합니다.
클러스터는 컨테이너들이 있는 관리도구 같은건가 봅니다.

2개의 노드를 갖는 클러스터를 만듭니다.
```bash
gcloud container clusters create hello-world \
                --num-nodes 2 \
                --machine-type n1-standard-1 \
                --zone us-central1-f
```

이제 컨테이너들을 관리하는 클러스터를 만들었고 포드를 만들 차례입니다!
쿠버네티스 포드는 관리 및 네트워킹 용도로 서로 연결된 컨테이너 그룹이고 하나 또는 여러 개의 컨테이너를 포함할 수 있다네요. 클러스터랑 다른 점은 서로 연결된 컨테이너 그룹이란건가요? 잘 모르겠네요.

이렇게 포드를 만듭니다.
```bash
kubectl run hello-node \
    --image=gcr.io/PROJECT_ID/hello-node:v1 \
    --port=8080
```

이렇게 hello-node 라는 배포 개체가 만들어졌습니다. 이제 이 포드를 세상에 노출시켜봅니다.

```bash
kubectl expose deployment hello-node --type="LoadBalancer"
```

신기하게 이렇게 ip 주소를 통해서 서버에 접속이 되네요.
<div>
<img class="aligncenter size-full" src="/image/kubernetes/1.png" alt="" /> </div>

마지막으로 서비스의 규모를 확장해봅니다.
어느 날 갑자기 어플리케이션에 더 많은 용량이 필요해졌다면, 복제 컨트롤러에 포드의 새로운 복제 여러 개를 관리하게 할 수 있습니다.
딱 한줄로요

```bash
kubectl scale deployment hello-node --replicas=4
```

이렇게 도커 이미지를 만들고 컨테이너 클러스터를 만들고 포드를 만들어 배포하고 확장해보았네요.
