---
id: 464
title: Simulated Annealing 최적화 알고리즘 코드 짜보기
date: 2017-11-29T14:06:45+00:00
author: JuHyung Son
layout: post
categories:
  - studying
tags:
  - 최적화
---
<h2>Simulated annealing</h2>

Simulated annealing은 최적해를 찾는 확률 모델입니다. 가장 단순한 "gradient-free" 최적화 모델이기도 하죠. Simulated annealing은 Annealing, 담금질에서 아이디어를 얻었습니다. 쇠를 담금질할 때, 온도가 높아 새빨간 철은 비교적 약간 충격이도 변형이 잘되고 온도가 내려갈수록 본래의 색을 띠며 단단해지며 만들고자하는 모습에 가까워집니다. Simulated annealing에서는 높은 온도로 시작해 온도를 떨어뜨리며 최적의 값, 담금질에서 만들고자 했던 쇠의 모습을 찾습니다.

온도가 높을 때는 조그만 확률에도 변화가 심하여 local minimum을 가뿐히 뛰어넘습니다. 온도가 낮아지면서 global minimum에 근접한 곳에 자리잡게 되고 global minimum에 정착합니다. 밑 그림은 global maximum을 찾는 과정입니다. simulated annealing의 코드를 봅시다.

<div align='center'> <img src="/wp-content/uploads/2017/11/SA_animation.gif" alt="" width="500" height="161" /> </div>

<h2>Code</h2>

Simulated annealing을 하려면 먼저 문제가 있어야하겠죠. 간단한 문제를 먼저 만듭시다. 문제는 간단한 TSP를 만듭니다. TSP는 Traveling Salesman Problem으로 간단한 소개는 다음 사이트를 보실 수 있습니다.

<a href="https://en.wikipedia.org/wiki/Travelling_salesman_problem">TSP 위키피디아</a>

<h3>TSP</h3>

```
class TravelingSalesmanProblem:

def __init__(self, cities):
self.path = copy.deepcopy(cities)

def copy(self):
"""현재 상태를 return"""
new_tsp = TravelingSalesmanProblem(self.path)
return new_tsp

@property
def names(self):
"""도시 이름 list를 return.
[("Atlanta", (585.6, 376.8)), ...] -> ["Atlanta", ...]
"""
names, _ = zip(*self.path)
return names

@property
def coords(self):
"""좌표를 return
[("Atlanta", (585.6, 376.8)), ...] -> [(585.6, 376.8), ...]
"""
_, coords = zip(*self.path)
return coords

def successors(self):
neighbors=[]
for i in range(len(self.path)):
temp = self.path.copy()
temp1 = temp[i-1]
temp2 = temp[i]
temp[i-1] = temp2
temp[i] = temp1
#name,_ = zip(*temp)
neighbors.append(temp)
return [TravelingSalesmanProblem(s) for s in neighbors]

Cost는 지점간의 직선거리로 한다.

def get_value(self):
total = 0
for i in range(len(self.path)):
temp1 = self.path[i-1]
temp2 = self.path[i]
temp1_coord = temp1[1]
temp2_coord = temp2[1]
dist = np.sqrt((temp1_coord[0]-temp2_coord[0])**2 + (temp1_coord[1] -temp2_coord[1])**2)
total += dist
return -total

```
<h3>Simulated annealing code</h3>
```
def simulated_annealing(problem, schedule):

stopping = True

current = problem
i=1
while stopping:
T = schedule(i)
next_step = problem.successors()
next_step = random.choice(next_step)
energy = next_step.get_value() - current.get_value()
if energy > 0:
current = next_step
else:
prob = random.binomial(1,np.exp(energy/T))
if prob > 0:
current = next_step
if T < 1e-10:
return current
i+=1
```

다음 온도와 온도의 감소 스케쥴을 정해줍니다. 온도는 exponential 로 감소하는 스케쥴로 지정합니다. 이 방법이 가장 일반적인 방법이고 다른 스케쥴을 짜도 가능합니다.

```
alpha = 0.95
temperature=1e4

def schedule(time):
return np.power(alpha, time) * temperature
```

미국 지도를 가지고 풀어본 TSP의 예.

<div align='center'>
<img class="aligncenter wp-image-471 size-full" src="/wp-content/uploads/2017/11/스크린샷-2017-11-29-오후-2.00.53-1.png" alt="" width="1018" height="563" /> </div>
