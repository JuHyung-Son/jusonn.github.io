---
title: BeautifulSoup를 이용한 블로그 크롤링
author: JuHyung Son
layout: post
tags:
   - crawling
   - python
categories:
  - Dev
---
<div align="center"> <img src="https://www.crummy.com/software/BeautifulSoup/bs4/doc/_images/6.1.jpg"/> </div>
이번에는 BeautifulSoup를 이용해 크롤링을 해봅니다.

BeautifulSoup은 HTML, XML 파일에서 데이터를 뽑아내는 파이썬 라이브러리입니다.


<a link="https://www.crummy.com/software/BeautifulSoup/bs4/doc/">BeautifulSoup</a>

라이브러리를 이용해 이 블로그의 포스트 주소들을 크롤링 해봅시다. 가장 먼저 BeautifulSoup에 HTML 파일을 전달해줘야 합니다. BeautifulSoup은 url에 접근하는 모듈을 없고 받아진 파일만을 다루기 때문에 웹으로부터 HTML 파일을 받아오는데는 `urllib`을 사용하여야 합니다.

그러니 먼저 제 블로그의 HTML을 받아온 후 BeautifulSoup에 파일을 넣고 type을 확인해봅시다.
```
r =urllib.request.urlopen("https://jusonn.github.io/blog").read()
soup = BeautifulSoup(r)
print(type(soup))
```
> <class 'bs4.BeautifulSoup'>

이제 soup으로 HTML을 읽어봅니다.
```
print(soup.prettify()[0:1000])
```
<div align="center"> <img src="/image/crawling/1.png"/> </div>

잘 들어 갔군요. 브라우저에서 페이지 소스보기를 하면 보이는 것과 같은 것입니다. 하지만 이 무지하게 긴 파일에서 무언가를 찾을 순 없겠죠. `div` 단위로 나눈 후 블로그 포스트 리스트만 뽑아와 봅니다.
```
letters = soup.find_all('div')
```
<div align="center"> <img src="/image/crawling/2.png"/> </div>

```
letters[9].find_all("a")
```
<div align="center"> <img src="/image/crawling/3.png"/> </div>

포스트들이 있는 부분이 잘 뽑아졌습니다.

여기서 바로 포스트 제목과 주소를 뽑아와도 됩니다. 하지만 전 이런 방법으로 {제목:주소} 형태의 데이터를 뽑아 볼 겁니다.
```
prefix = 'https://jusonn.github.io'
lobbying = {}
for element in letters[9]:
    try:
        element = element.find_all('a')
        for elem in element:
            lobbying[elem.get_text()] = prefix + elem['href']
    except:
        pass
```
<div align="center"> <img src="/image/crawling/5.png"/> </div>

성공적입니다! 바로 접속이 가능한지 확인해봅시다.

<div align="center"> <img src="/image/crawling/6.png"/> </div>
