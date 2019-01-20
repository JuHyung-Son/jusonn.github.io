---
title: 쿠버네티스를 통한 클라우드 조정
date: 2019-01-20
author: JuHyung Son
layout: post
tags:
  - kubernetes
---

구글 스터디잼 쿠버네티스 입문 과정을 수강 중
구글 클라우드를 통해 자습형 실습을 해보았다.

사실 저번부턴 뭐가 진행되는지 이해 못하고 따라 치기만 하고 있다.

이번에 학습한 내용은
1. 쿠버네티스 엔진
2. 쿠버네티스 엔진을 사용해 쿠버네티스 클러스터를 감독
3. kubectl 을 이용해 도커 컨테이너를 배포하고 관리
4. 쿠버네티스 배포 및 서비스를 사용해 하나의 어플리케이션을 마이크로 서비스로 나눔.

스터디잼에서 제공하는 샘플 코드로 진행중
```bash
git clone https://github.com/googlecodelabs/orchestrate-with-kubernetes.git
```

쿠버네티스를 이용해 nginx 컨테이너의 단일 인스턴스를 실행해보고 봐봅니다.
그리고나서 이 컨테이터를 쿠버네티스 밖으로 노출시켜봅시다.
```bash
kubectl run nginx --image=nginx:1.10.0
kubectl get pods
kubectl expose deployment nginx --port 80 --type LoadBalancer
```

컨테이너가 외부에 노출되었는지 보려면
```
kubectl get services
```
로 서비스를 나열해 봅니다.
노출되었다면 external IP 에 무언가가 나오겠죠.

## pod

포드는 쿠버네티스의 핵심 요소로 하나 이상의 컨테이너가 포함된 집합을 의미합니다. 일반적으로는 종속도가 높은 여러 컨테이너가 있을 떄 컨테이너들을 단일 포드로 패키징한다네요.
이 포드를 만듭어봅니다.

포드는 구성 파일을 사용해 만들 수 있고 구성파일은 이렇게 생겼습니다.

```bash
apiVersion: v1
kind: Pod
metadata:
    name: monolith
    labels:
        app: monolith
spec:
    containers:
        - name: monolith
          image: kelseyhightower/monolith:1.0.0
          args:
            - "-http=0.0.0.0:80"
            - "-health=0.0.0.0:81"
            - "-secret=secret"
          ports:
            - name: http
              containerPort: 80
            - name: health
              containerPort: 81
          resources:
            limits:
                cpu: 0.2
                memory: "10Mi"
```

여기서 알 수 있는 건
- 포드는 하나의 컨테이너(모놀리식)으로 구성됩니다.
- 시작할 때 몇 개의 인수를 컨테이너에 전달합니다.
- http 트래픽을 위해 포트 80을 엽니다.

이제 위의 포드를 만들어 봅니다.

```bash
kubectl create -f pods/monolith.yaml
kubectl get pods
```

포드는 기본적으로 클러스터 외부에 도달할 수 없습니다. 그래서 포트 포워딩을 이용해 로컬 포트를 방금 전에 만든 포드의 포트로 매핑해야합니다. 그러면 매핑된 포트를 이용해 포드와 통신을 할 수 있죠.

