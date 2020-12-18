# introduction-tensorflow

## WK1. A new programming paradigm

### Introduction 
전통적인 방식의 프로그램과 머신러닝을 사용한 프로그래밍(?)에 있어서 기본 패러다임이 바뀐 부분에 대해 강조하여 설명 하고 있다. 

### 전통적인 프로그래밍 Vs 머신러닝
- 전통적인 프로그램: 입력 데이터와 규칙을 바탕으로 프로그래밍 하면 정답을 출력
- 머신러닝: 입력데이터와 정답(영어로 Answer, 코딩에선 라벨링)을 입력하여 머신러닝 모델링 하면 규칙을 출력. 

https://blog.naver.com/reisei11/221632271189


### 즉!!! 
이러한 규칙은 패턴을 찾는 과정이다. 

### 예시: 사람의 활동을 프로그래밍 한다면,  

- 전통적인 프로그래밍의 경우 활동을 나타내는 동작은 "속도"를 기준으로 쉽게 판단 가능하다. 그렇다면, 걷다 -> 뛰다 -> 자전거를 타다는 구현 가능하지만 골프는??? 불가능 하다. 

- 반면에, 머신러닝의 경우 걷다의 데이터와 그 행동 양상에 대한 라벨링(정답), 뛰다의 데이터와 그 행동양상에 대한 라벨링(정답)을 주면 그러한 데이터를 통해 "패턴", "규칙"을 찾아내고 이런식으로 코딩하면 골프역시 구현이 가능하다. 

- 그리고, 패턴을 인식하기 가장 좋은 방법은 뉴런을 사용하는 것!


### 머신러닝의 구현방법은?
- Tensorflow와 Tensorflow 기반 API인 Keras를 이용해서 구현해 본다. 
- 특별히, 뉴런의 구현을 모델이라고 하는데, 
    - 뉴런들의 층(layer)은: ```tf.keras.models.Sequential()``` Sequential()로 구현한다. 
    - 뉴런은: ```tf.keras.layers.Dense()``` Dense로 구현한다. 


- 기본 workflow

1. 모듈 임포트
2. "모델 구현" - 코드 버전
   - model = Sequential()
   - model.compile(optimizer, loss function)
   - model.fit(x, y, epoch)
   - model.predict
3. 모델 구현 - 말 버전
    - 모델 설계 (build neural network, "build NN model")
    - 모델 컴파일 (설계의 일부, optimizer와 loss function 설정, "Set a optimizer & loss function")
    - 모델 훈련 (train NN model, fit a NN model, 훈련-입력데이터, 훈련-라벨링, 훈련횟수를 인자로 넣는다.)
    - 예측 (test data입력으로 넣고 결과값을 출력)

```
import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(xs, ys, epochs=10)

print(model.predict([10.0]))
```

### 조금 생각해 봐야 할 문제
- Model & Neural Network
- optimizer
- loss function

Now we compile our Neural Network. When we do so, we have to specify 2 functions, a loss and an optimizer.

If you've seen lots of math for machine learning, here's where it's usually used, but in this case it's nicely encapsulated in functions for you. But what happens here — let's explain...

We know that in our function, the relationship between the numbers is y=2x-1.

When the computer is trying to 'learn' that, it makes a guess...maybe y=10x+10. The LOSS function measures the guessed answers against the known correct answers and measures how well or how badly it did.

It then uses the OPTIMIZER function to make another guess. Based on how the loss function went, it will try to minimize the loss. At that point maybe it will come up with somehting like y=5x+5, which, while still pretty bad, is closer to the correct result (i.e. the loss is lower)

It will repeat this for the number of EPOCHS which you will see shortly. But first, here's how we tell it to use 'MEAN SQUARED ERROR' for the loss and 'STOCHASTIC GRADIENT DESCENT' for the optimizer. You don't need to understand the math for these yet, but you can see that they work! :)

Over time you will learn the different and appropriate loss and optimizer functions for different scenarios.


## WK2. Introduction to Computer Vision

### Introduction 
실전 문제를 풀어 보자. Computer vision을 이용해서 분류 문제를 풀어보자~ 

### 차이점

- 데이터 : 28* 28 픽셀로 된 흑백 이미지, 70,000개 데이터, 10종류.
```
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) 
= fashion_mnist.load_data() 

```

- layer : 3층
```
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)])
```


### Advanced Workflow

1. 모듈 임포트
2. 데이터 로드
  - fashion_mnist = keras.datasets.fashion_mnist
  - normalization (정규화, 0 ~ 1사이 값으로)
2. "모델 구현" - 코드 버전
   - model = Sequential([
       keras.layers.Flatten(input_shape=(28, 28)),
           keras.layers.Dense(128, activation = tf.nn.relu),
           keras.layers.Dense(10, activation = tf.nn.softmax)],
   ])
   - model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
   - model.fit(training_images, training_labels, epochs=5)
   - model.evaluate(test_images, test_labels)
   - model.predict
3. "모델 구현" - 말 버전 생략
4. WK1과의 차이점
    - 데이터 로드,3layer (비디오에서 언급)
    - layer중간에 활성화 함수를 사용함
    - optimizer, loss function 다름.
    - model.evaluate 추가됨.
5. Call back 함수
    - 훈련 횟수를 조정하기 위함. 
    - def on_epoch_end(self, epoch, logs={})

### 몇가지 연습. 
2. 중간 뉴런의 갯수를 128개가 아닌 1024 개로 한다면?
3. Flatten이 없다면?
4. output 뉴런이 10이 아닌 다른 숫자라면?
5. 중간 뉴련의 층을 늘리고 그 숫자를 각각 512, 256개로 한다면?
6. epoch를 10회에서 30회로 변경한다면? 
7. 정규화(nomarlize)를 하지 않는다면?
8. 콜백함수...?

### 더 생각해 봐야할 문제
- 콜백함수
- optimizer, loss function


# 마치며, 

- 저는 패러다임의 변화 라는 키워드에 대해 많이 생각하려고 노력했습니다.
- 왜 패턴을 인식하고 패턴을 인식할 때 좋은 기법이 뉴런이라는 점에 주목했습니다. 
- optimizer, lossfunction, Dense등 구현에 대한 부분은 슥슥 넘겼습니다. 
- Callback 함수를 이용해 훈련횟수를 조정한다는 사실은 처음 봤고, 구현 원리는 이해 불가 이는 공부가 더 필요합니다. 
- 퀴즈를 푸는건 어렵지 않았으나 숙제를 통과하는것은 실패 했습니다. 

- 추가 동영상을 들었고 아래 3가지는 특히 더 도움이 되었습니다.  

https://youtu.be/VwVg9jCtqaU

https://youtu.be/BKj3fnPSUIQ

https://youtu.be/aNrqaOAt5P4
