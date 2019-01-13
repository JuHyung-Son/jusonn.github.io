
---
title: 자바스크립트 promises, async, await 이 뭘까
date: 2019-01-13
author: JuHyung Son
layout: post
tags:
  - js
---

자바스크립트에 들어서기 시작하는 저는 promise, async가 계속 헷갈립니다.
그러니 한번 정리를 해볼까요.

## Callback

자바스크립트에서의 많은 실행은 비동기적으로 일어납니다. 함수가 지금 당장 끝나는게 아니라 나중에 끝나는게 대부분이란 말이죠.

아래처럼 스크립트는 로드하는 함수가 있다고 해봅니다. 

```javascript
function loadScript(src) {
  let script = document.createElement('./script.js');
  script.src = src;
  document.head.append(script);
}
```

만약 ```'./script.js'``` 에 ```newFunction()``` 이라는 함수가 있었다면 다음 코드는 주석과 같은 에러를 뱉겠죠.

```javascript
loadScript('./script.js');

newFunction(); // newFunction is not defined
```

'./script.js' 가 전부 로드되기 전에 내부함수가 실행되었기 때문입니다. 이 문제를 해결하기 위해 callback 함수를 추가해봅니다.

```javascript
function loadScript(src, callback) {
  let script = document.createElement('script');
  script.src = src;

  // 스크립트가 로드되면 callback을 실행
  script.onload = () => callback(script);
  document.head.append(script);
}
```

이제 다음처럼 ```newFunction()```을 실행할 수 있네요.

```javascript
loadScipt('./script.js', function() {
  // callback은 스크립트가 로드된 후 실행됨.
  newFunction();
})
```

이렇게 callback 함수를 이요하는 방식을 'callback-based' 비동기 방식이라고 합니다. 비동기적으로 작동하는 모든 비동기 함수는 callback 함수를 가져야하죠. 이런식으로 여러개의 비동기 함수를 쌓을 수도 있습니다.

```javascript
loadScript('./script.js', function(script) {
  loadScript('./script.js', function(script) {
    loadScript('./script.js', function(script) {
      // 반복
    })
  })
})
```

그리고 에러처리는 이런 방식으로 할 수 있죠.

```javascipt
function loadScript(src, callback) {
  let script = document.creteElement('scipt');
  sciprt.src = src;

  script.onload = () => callback(null, script);
  script.onerror = () => callback(new Error('Scipt load error'));

  document.head.append(script);
}
```

실제로 보통 이렇게 사용합니다.

```javascript
loadScript('./script.js', function(error, script) {
  if (error) {
    // 에러
  } else {
    //스크립트 로드됨
  }
})
```

이제 비동기 코드가 매우 잘 작동합니다. 하지만 한 함수안에 여러개의 callback 함수가 쌓인다면 가독성이 매우 떨어져죠. 본문에서는 callback hell 에 빠졌다고 하네요. 이러한 피라미드 모양의 코드를 피하기 위해 함수를 따로 적어줘도 코드는 아래위로 길어지게 됩니다. 다행히 promise 라는 비동기 코딩 방식이 존재합니다. 

## Promise

당신이 잘나가는 가수라고 해봅시다. 그리고 팬들은 집앞에 몰려와 밤낮으로 24시간 다음 앨범에 대해 묻고 있습니다.
당신은 마음을 가다듬고 팬들에게 발매일이 확정나면 알려줄테니 이메일을 적고 구독히리고 합니다. 꼭 알려주겠다고 약속(promise) 하고요.
이것이 promise 입니다.

생산 코드는 시간이 걸리는 어떤 작업을 합니다. 가수처럼요.
소비 코드는 생산 코드의 결과가 나온다면 그것을 알고 싶어합니다. 팬들처럼요.
promise 는 생산 코드와 소비 코드에 연결된 특별한 자바스크립트 객체입니다. 이게 바로 구독이죠.

promise 객체는 다음처럼 보여요.

```javascript
let promise = new Promise(function(resolve, reject) {
  // 생산 코드 작업
})
```

```new Promise``` 안에 익명함수는 excutor 라고 불립니다. promise 객체가 생성되면 자동으로 실행되는 함수예요. 이 promise 객체는 다음과 같은 특성을 내부적으로 갖습니다.

- state -> 초기값은 'pending', 이후 'fulfilled' 혹은 'rejected'로 바뀜.
- result -> 함수의 리턴값, 초기값은 undefined

excutor 가 완료되면 결과에 따라 다음 둘 중 하나를 실행합니다.

- resolve(value): excutor가 성공적으로 끝남.
  - state -> 'fulfilled'
  - result -> value
- reject(error): 에러가 발생했다는 걸 알림.
  - state -> 'rejected'
  - result -> error

간단한 예를 봅니다.

```javascript
let promise = new Promise(function(resolve, reject) {
  setTimeout( () => resolve('done!'), 1000);
});
```

여기선 익명함수인 excutor가 자동으로 실행되고 에러가 없으면 1초후 ```resolve('done!')``` 이 실행됩니다.
promise는 자바스크립트 내부에 이미 구현되어 있으므로 몇가지 규칙을 가집니다.

1. result나 error는 딱 하나 뿐이다.
2. Error 객체로 reject 하는 것이 좋다.
3. state와 result는 내부 객체다.

여기까지가 생성 코드, 즉 가수가 하는 것입니다. 팬의 입장이 되어 promise를 사용해봅시다.
promise 객체가 만들어졌다면 .then과 .catch로 결과를 얻을 수 있습니다.

.then : promise가 resolved 되고 결과를 받았을 떄 실행되는 함수 인자를 가지거나 ,rejected되고 에러를 받았을때 실행되는 함수 인지를 가질 수 있습니다.
.catch : 에러가 발생했을시 실행하는 함수를 인자로 가집니다. 
예시 

```javascript
let promise = new Promise(function(resolve, reject) {
  setTimeout( () => resolve('done'), 1000);
});

promise.then(alert);
```

이제 위의 ```loadScript``` 함수를 promise 를 사용해 구현해봅니다.

```javascript
// promise 생성
function loadScript(src) {
  return new Promise(function(resolve, reject) {
    let script = document.createElement('script');
    script.src = src;

    script.onload = () => resolve(script);
    script.onerror = () => reject(new Error("script load error"));

    document.head.append(script)
  })
}

// 사용
let promise = loadScript('www.naver.com')

promise.then(
  script => alert('loaded');
  error => alert('error');
);
promise.then(script => alert('do something'));
```

이렇게 promise 를 사용하면 자연스런 순서대로 코드가 동작합니다. 또 promise.then() 을 이후에도 얼마든지 부를 수 있죠. 하지만 함수가 실행되기전 함수의 결과값으로 어떤 작업을 비동기로 할 지 알아야하고 딱 하나의 callback 만 가질 수 있다는 단점도 있습니다. 

## fetch

자바스크립트 내장함수인 fetch 를 봅시다. 주로 서버로부터 뭔가를 요청할 때 자주 쓰입니다. 
아래와 같이 쓰입니다.

```javascript
// Make a request for user.json
fetch('/article/promise-chaining/user.json')
  // Load it as json
  .then(response => response.json())
  // Make a request to github
  .then(user => fetch(`https://api.github.com/users/${user.name}`))
  // Load the response as json
  .then(response => response.json())
  // Show the avatar image (githubUser.avatar_url) for 3 seconds (maybe animate it)
  .then(githubUser => {
    let img = document.createElement('img');
    img.src = githubUser.avatar_url;
    img.className = "promise-avatar-example";
    document.body.append(img)

    setTimeout(() => img.remove(), 3000);
  })
  .catch(console.log(err))
```