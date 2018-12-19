# Deepest Quest  1

박진호 (jinh0park@naver.com)

---

## (b) 추가 문제 (grad cam)

### Summary

#### Correct Prediction Saliency Map

![0](https://user-images.githubusercontent.com/39009836/50231251-b0386b80-03f1-11e9-944b-6da4fcaec2eb.png)

**분석**:  
세 동물의 Prediction 결과를 가져왔다. 주로 Saliency Map이 얼굴 부분에 하나, 몸통 부분에 하나가 나타났음을 볼 수 있다. 직관적으로 얼굴은 동물을 구분하는 가장 큰 특징 중 하나이며, 몸통은 그 두 번째 요소라고 생각해 봤을 때 납득할 만한 결과임을 알 수 있다.

---

#### Incorrect Prediction Saliency Map

![1](https://user-images.githubusercontent.com/39009836/50231223-a0b92280-03f1-11e9-95c4-4020b6fc936a.png)


**분석**:  
세 동물의 Prediction 결과를 가져왔다. 순서대로 각각의 Prediction이 틀린 이유를 꼽아보자면 다음과 같다.

1. 얼굴 + 몸통의 조합으로 인식을 하는데, 얼굴만 있기 때문에 얼굴 자체를 얼굴 + 몸통으로 인식하였고, 따라서 말의 얼굴이 개가 되었다(실제로 눈을 흐릿하게 멀리서 이 사진을 보면 갈색에 흰색 줄무늬가 있는 강이자로 보이기도 한다).
2. 새 앞에 열매같이 생긴 무언가가 가로막고 있다. 아마도 모델은 이 물체까지 개체로 인식하고 잘못된 예측을 내놓았을 것이다.
3. Saliency Map이 오른쪽 아래에 집중되어 있고, 이를 Cat으로 인식했다. 오른쪽 아래의 검은 부분이 있고, 그것을 고양이의 얼굴 부분이라고 판단했을 가능성이 있다고 생각한다.


---

### 비고

마지막과 그 다음 Covolution Layer의 Feature Map Size가 각각 4*4, 8*8로, 이를 이용하여 GradCam을 그리기에는 Resolution이 너무 낮다는 문제가 있었다. 따라서 그 마지막에서 세 번째 Conv Layer를 Grad Cam을 그리는데 이용하였다.

그럼에도 불구하고 Resolution이 낮아, Saliiency Map을 만들 때 효과적인 Visualization을 위해 Blur처리를 하였다.
