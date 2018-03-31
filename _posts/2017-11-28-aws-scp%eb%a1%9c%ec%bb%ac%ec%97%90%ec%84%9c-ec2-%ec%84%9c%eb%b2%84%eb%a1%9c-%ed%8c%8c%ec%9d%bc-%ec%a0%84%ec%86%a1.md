---
id: 458
title: AWS scp,로컬에서 EC2 서버로 파일 전송
date: 2017-11-28T00:55:18+00:00
author: JuHyung Son
layout: post
permalink: '/aws-scp%eb%a1%9c%ec%bb%ac%ec%97%90%ec%84%9c-ec2-%ec%84%9c%eb%b2%84%eb%a1%9c-%ed%8c%8c%ec%9d%bc-%ec%a0%84%ec%86%a1/'
dsq_thread_id:
  - "6313621152"
image: /wp-content/uploads/2017/11/texture-1444488.jpg
categories:
  - Dev
---

EC2를 사용한다면 보통 로컬에 있는 파일을 올려서 작업을 해야할 때가 대부분이다. 나 같은 경우는 딥러닝에 필요한 학습데이터를 로컬에서 서버로 옮긴 후 학습을 시킨다. scp라는 걸 모르기 전까진 주피터에 접속해서 일일이 upload를 했다는...  앞으론 다음과 같이 파일을 전송하자.

맥을 기준으로 쓴다.

<h2>서버 접속</h2>
서버는 편한 방식으로 접속하자.

```
ssh -i [pem file] [user id]@[ec2 public IP]:~/[transfer adress]
#예시
ssh -i Desktop/amazon/juhyung.pem ubuntu@~~~~~:~/

```
<h2>파일 전송</h2>
다음 새 터미널을 열어서 파일을 전송한다.

```
#파일 전송시
scp -i [pem file] [upload file] [user id]@[ec2 public IP]:~/[transfer address]
#예시
scp -i Desktop/amazon/juhyung.pem Desktop/pant.py ubuntu@~~~~:~/
#폴더 전송시
scp -i [pem file] -r [upload folder] [user id]@[ec2 public IP]:~/[transfer address]
#예시
scp -i Desktop/amazon/juhyung.pem -r Desktop/example ubuntu@~~~~:~/

```
<h2>EC2 서버에서 로컬로 파일 전송</h2>
단지 위에서 방향만 반대로 바꾸면 된다.
```
scp -i [pem file] [user id]@[ec2 public id]:~/[transfer address] [local address]
```
<div class="grammarly-disable-indicator"></div>
<div class="grammarly-disable-indicator"></div>
