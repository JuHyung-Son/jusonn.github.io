---
id: 307
title: 개미 집단 알고리즘,최적화, ACO
date: 2017-11-12T16:21:17+00:00
author: JuHyung Son
layout: post
permalink: '/%ea%b0%9c%eb%af%b8-%ec%95%8c%ea%b3%a0%eb%a6%ac%ec%a6%98-ant-colony-optimization-algorithms/'
dsq_thread_id:
  - "6278647869"
image: /wp-content/uploads/2017/11/cropped-IMG_5429-1-250x250.jpg
tags:
   - optimization
   - 최적화
categories:
  - Studying
---
<h2>Ant Colony optimization algorithms</h2>
<img class="aligncenter size-medium" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Safari_ants.jpg/440px-Safari_ants.jpg" width="440" height="292" />

Ant colony optimization(ACO)는 계산문제를 푸는 확률적 해법 중 하나로 그래프에서 최적의 경로를 찾는 데 쓰인다. dijkstra 같은 polynomial time을 가지는 알고리즘은 노드가 증가함에 따라 경로 탐색에 걸리는 시간이 오래 걸리는 단점이 있다. 그렇기 때문에, heuristic search로 접근하여 최적의 경로가 아니더라도 빠른 시간 내에 적절한 경로를 찾는 방법을 사용할 필요가 있다. ACO, A*와 같은 알고리즘이 대표적이다.
개미 알고리즘은 개미의 집단행동에서 아이디어를 가져왔다. 개미가 개미집에서 먹이로의 경로를 찾을 때, 먹이를 발견한 첫 개미가 페로몬을 남기며 집으로 돌아온다. 다음 개미는 역시 페로몬을 뿌리면서 앞선 개미의 페로몬을 따라 먹이를 찾는데, 중간중간 희미한 페로몬 때문에 경로가 조금씩 바뀌기도 하면서 여러 가지 경로가 쌓인다. 페로몬은 시간이 지나면서 증발하고 결국 최적의 경로에만 페로몬이 남게 된다.
개미 알고리즘은 여러 변형된 알고리즘들이 있는데 이것들을 Ant colony algorithm family라고 한다.

- Elitist ant system
매 반복마다 가장 좋은 솔루션에만 페로몬을 남긴다.

- Rank based ant system
모든 솔루션을 cost에 따라 랭크를 매긴다. 그 랭크에 따라 각각 다른 페로몬이 솔루션에 업데이트된다. 가장 낮은 코스트에 페로몬을 더 남기고 높은 코스트에 페로몬을 덜 남긴다.

- Continuous orthogonal ant colony
이 방법은 개미들이 최적의 솔루션을 협력하고 효과적으로 찾게 해준다. 이 알고리즘으로, 비슷한 곳에 있는 개미들은 정해진 지역을 빠르고 효과적으로 탐색할 수 있다.

- Recursive ant colony optimization
개미 시스템의 recursive한 형태이다. 모든 탐색 지역을 여러개의 작은 탐색 지역으로 나누고 각각의 작은 탐색 지역에서 솔루션을 찾는다. 다음 모든 작은 탐색 지역에서의 솔루션을 비교해 가장 좋은 몇 가지 솔루션을 다음 검색에 사용한다. 각각의 선택된 작은 탐색 지역은 또 다시 더 작은 탐색 지역으로 나뉘어지고 다시 탐색하며 이 과정을 반복한다.
<h2>Convergence</h2>
Ant colony family 중 몇몇은 그것이 결국 수렴한다는 것이 증명가능하다. (제한된 시간내에 global optimum을 찾을 수 있다는 것.) 그래프 기반의 개미 알고리즘의 수렴성은 2000년에 처음 증명되었다. 대부분의 메타휴리스틱 모델처럼, 이 알고리즘의 이론적 수렴속도를 찾는 것은 매우 어렵다.
<h2>Pseudo code</h2>
```
procedure ACO_MetaHeuristic
while(not_termination)
generateSolutions()
daemonActions()
pheromoneUpdate()
end while
end procedure
```
<h2>Edge selection</h2>
개미 알고리즘에서 개미는 간단한 계산 agent이다. 이 개미은 직접 돌아다니며 계속해서 솔루션을 찾는다. 각각의 개미k는 현재 상태에서 실행가능한 솔루션 A_k(x)를 계산한 다음, 그것들 중 한 길로 확률에 따라 이동한다. 개미가 x에서 y로 이동할 확률은 두가지 값의 조합으로 결정된다. <em>the attractiveness</em>(heuristic으로 계산된 그 움직임의 바람직함?)와 <em>the trail level</em>(이전의 움직임이 얼마나 효율적이었는지 알려주는)이다.
Trail은 해당 움직임이 얼마나 바람직한지를 알려주는 값이다. trail은 개미가 자신의 솔루션은 찾았을 때 업데이트되고, 그 솔루션이 얼마나 좋았는지에 따라 값이 증가하고 감소한다. 개미가 뿌린 페로몬 양으로 보면 된다.
<h2>Applications</h2>
개미 알고리즘은 여러 조합 최적화 문제에 적용된다. 물론 경로 찾기 모델에도 사용가능하다. 또 dynamic problem에 맞게 개조도 많이 되었다. 예, real variable, stochastic problem, multi-targets etc. 또 Travelling salesman problem에서도 최적에 가까운 솔루션을 찾기도 한다. 유동적인 그래프에서는 Simulated annealing과 genetic algorithm보다 장점이 많다고 알려져 있다. 한 예로, 실시간으로 실행하고 바로 적용이 가능한데 이것은 network routing과 도시 이동 시스템에서 관심이 큰 관심거리이기도 하다.

가장 초기의 개미 알고리즘은 TSP문제를 해결하는 것이 목표였다. 이 알고리즘은 매우 간단하고 개미라고 불리는 것들을 이용한다. 각 개미는 가능한 경로를 통해 도시를 방문한다. 이 때 개미는 다음의 규칙을 가지고 도시 사이를 이동한다.

<img class="aligncenter size-medium" src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Aco_TSP.svg/1200px-Aco_TSP.svg.png" width="1200" height="450" />
1. 모든 도시를 딱 한번만 방문한다.
2. 먼 도시로 갈 확률은 낮다.
3. 두 도시 사이에 페로몬 농도가 진하다면 그 연결된 도시로 갈 확률이 높다.
4. 만약 개미가 간 경로가 짧다면, 개미가 지나간 모든 길에 페로몬을 더 뿌린다.
5. 매 반복마다 페로몬은 일정량 증발한다.

위 방식으로 그림과 같은 방법을 찾아낸다.  다음에는 개미 알고리즘을 코드로 짜보면서 적용하겠다.

<a href="https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms">위키보기</a>
<div class="grammarly-disable-indicator"></div>
