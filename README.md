# Deepest Quest  1

박진호 (jinh0park@naver.com)

---

##사용 프레임 워크:
TensorFlow v1.12.0

각 문제에 대한 보다 상세한 설명은 `prob_a.md`, `prob_b.md`에 적혀 있습니다.

##Project Tree:

    root
    │  .gitignore
    │  gradcam.ipynb
    │  gradcam_neat.ipynb
    │  new.md
    │  quest_final.ipynb
    │  README.md
    │  read_stl10_file.py
    │  requirements.txt
    │
    ├─data
    │  └─stl10_binary
    │          class_names.txt
    │          fold_indices.txt
    │          test_X.bin
    │          test_y.bin
    │          train_X.bin
    │          train_y.bin
    │
    └─model
        checkpoint
        final_model.data-00000-of-00001
        final_model.index
        final_model.meta

(data와 model folder는 용량 문제로 ignore하였습니다. model은 압축하여 메일에 첨부하였습니다.)

## 실행 방법

0. `pip install -r requirements.txt` 명령으로 Dependency를 설치한다.
1. stl-10 dataset을 `data/stl10_binary` folder에 복사한다.
2. `quest_final.ipynb` 파일을 실행하여 model을 train한다
3. `model` 폴더에 train된 모델이 저장된다.
4. `gradcam_neat.ipynb`를 실행하여 grad-Cam을 그린다.

## 각 파일 설명:

- `quest_final.ipynb`: 모델 정의 및 훈련 + 검증 코드
- `read_stl10_file.py`: stl-10 dataset을 쉽게 extract 해주는 코드(출처: [Github Repo](https://github.com/mttk/STL10/blob/master/stl10_input.py)
- `gradcam.ipynb`: Pre-trained model을 불러와 gradcam을 그려주는 코드
- `gradcam_neat.ipynb`: `gradcam.ipynb`의 보다 깔끔한 버전
