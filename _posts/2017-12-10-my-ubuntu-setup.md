---
id: 491
title: 나의 우분투 개발환경
date: 2017-12-10T21:01:04+00:00
author: JuHyung Son
layout: post
dsq_thread_id:
  - "6340699239"
tags:
   - 우분투
   - 개발환경
categories:
  - Dev
---

처음에는 잘 몰랐는데 개발환경 세팅하는게 상당히 골치아프네요.. 원래 맥에서는 그냥 주피터 하나만 설치하면 끝이었는데 말입니다.

제가 우분투에서 어떤 개발환경을 맞추는 지 저장용으로 작성합니다.

<ol>
 	<li>vim</li>
</ol>
설치

```
sudo apt-get install vim
```

vim 환경설정

다음과 같이 파일을 엽니다.

```
vi ~/.vimrc
```

다음을 작성하고 저장한 후 나갑니다 :wq
```
<pre class="">" 자동 문법 강조
syntax on

"color테마를 적용
":colorscheme spectro   "테마를 설치할 경우에 적용가능하다.

set nocompatible "Vi 와의 호환성을 없애고, Vim 만의 기능을 쓸 수 있게 함.
set hi=1000 "명령어 기록을 남길 갯수 지정
set bs=indent,eol,start "백스페이스 사용

"encoding setting
set enc=utf-8
set fenc=utf-8
set fencs=utf-8,cp949,cp932,euc-jp,shift-jis,big5,ucs-2le,latin1


set cindent "C언어 자동 들여쓰기 옵션
set autoindent " 자동 들여쓰기 옵션
set smartindent

set tabstop=2    "탭에 대한 공백 크기 설정
set shiftwidth=2   "autoindent 옵션이 존재할때 후향 탭의 공백의 수를 정의
set nu      "라인 번호
set hls     "검색어 강조

"프로그램 시작시 플러그인 로드
set lpl

" 괄호 자동 완성 후 입력모드로 전환
map! () ()i
map! (); ();hi
map! [] []i
map! {} {}i
map! {}; {};iO
map! <> <>i
map! '' ''i
map! "" ""i

"vim plug-in on
filetype plugin on
</pre>
```
2. gdb 설치



```
sudo apt-get install gdb
```

3. 아나콘다 설치

<a href="http://www.anaconda.com">www.anaconda.com</a>

위 사이트에서 설치한 후

bash Anaconda2-4.2.0-Linux-x86_64.sh로 설치 진행, 설치 완료 후 환경 설정

1) cd ~

2) gedit .bashrc

3) export PATH=/home/사용자이름/anaconda2/bin:$PATH 입력 후 저장

4) source .bashrc

이후 conda 로 설치 확인, pip3 설치

4.guake terminal 설치 후 자동실행 설정
```
sudo apt-get install guake
```

5. Atom 설치
<pre class=""><code class="language-obj-c" data-lang="obj-c">sudo add-apt-repository ppa:webupd8team/atom
sudo apt-get update
sudo apt-get install atom</code></pre>
&nbsp;

6. 간단한 꾸미기

plank 설치, gnome tweak tool, paper 테마 설치

```
sudo apt-get install plank
sudo apt-get install gnome-tweak-tool
sudo add<span class="token se_code_operator">-</span>apt<span class="token se_code_operator">-</span>repository ppa<span class="token se_code_punctuation">:</span>snwh<span class="token se_code_operator">/</span>pulp

sudo apt<span class="token se_code_operator">-</span><span class="token se_code_keyword">get</span> update

sudo apt<span class="token se_code_operator">-</span><span class="token se_code_keyword">get</span> install paper<span class="token se_code_operator">-</span>icon<span class="token se_code_operator">-</span>theme paper<span class="token se_code_operator">-</span>gtk<span class="token se_code_operator">-</span>theme
```

&nbsp;

바탕화면 --&gt; 벡터아트

<a href="http://wallpaperswide.com/vector_art-desktop-wallpapers">벡터아트 이미지 사이트 보기</a>
