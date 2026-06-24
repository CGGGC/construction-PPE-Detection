# 🚧 공사 현장 안전 장비 착용 상태 탐지 (Construction Safety Detection)

## 📌 프로젝트 개요

본 프로젝트는 공사 현장의 CCTV 및 사진 데이터를 활용하여, 작업자의 안전 장비(안전모, 안전조끼) 착용 여부를 실시간으로 판별하는 객체 탐지(Object Detection) AI 모델을 개발하고 비교 분석하는 프로젝트입니다.

다양한 환경과 객체 크기에 대응하기 위해 최신 YOLO 모델과 Faster R-CNN 모델을 학습시키고, 이를 기반으로 최적화된 커스텀 모델을 설계하여 성능을 검증합니다.

## 🛠 기술 스택

* Language: Python

* Framework & Library: PyTorch, Ultralytics(YOLO), Torchvision, Pillow, OpenCV

* Environment: Google Colab, Jupyter Notebook

## 📁 프로젝트 구조

1. [01] Data Preprocessing : 데이터 정제 및 모델별 라벨 포맷 변환

2. [02] YOLO & R-CNN Models : Baseline 모델 학습 및 평가

3. [03] Custom Model : 구조 개선을 통한 커스텀 객체 탐지 모델 설계

4. [04] Evaluation & Analysis : 모델별 성능 비교 및 결과 분석


## 📂 01. 데이터 전처리 (Data Preprocessing)

서로 다른 구조(YOLO, Faster R-CNN)를 가진 두 객체 탐지 모델의 입력 규격에 맞춰 원본 데이터를 정제하고 변환하는 데이터 파이프라인을 구축했습니다.

* 데이터셋 구성: Roboflow Construction Safety Dataset (클래스: 헬멧 착용/미착용, 조끼 착용/미착용, 사람)

* 이미지 정규화: 다양한 해상도의 이미지를 640x640 사이즈로 일괄 리사이징(Pillow 활용).

* YOLO 데이터셋 세팅: 텍스트(.txt) 기반의 정규화된 바운딩 박스(0~1 사이 상대값)를 구성하고, 학습 자동화를 위한 data.yaml 생성.

* R-CNN 데이터셋 세팅: YOLO 포맷을 Faster R-CNN용 COCO 포맷(.json)으로 직접 변환. 상대 좌표를 절대 픽셀 좌표(x_min, y_min, width, height)로 매핑하는 알고리즘 구현.


## 📂 02. Baseline 모델 학습 및 비교 (YOLOv8 vs Faster R-CNN)

동일한 전처리 데이터를 바탕으로, 객체 탐지 분야의 대표적인 두 모델 아키텍처를 학습시키고 장단점을 대조했습니다.

### YOLOv8 (1-Stage Detector)

* 특징: 이미지를 한 번의 네트워크 통과(Backbone -> Neck -> Head)만으로 객체 위치와 클래스를 동시에 예측합니다.

* 장점: 연산 속도가 매우 빨라, 공사 현장 CCTV 모니터링 등 실시간 탐지(Real-time Detection)에 강점이 있습니다.

* 사용 목적: 실시간 처리 환경에서의 Baseline 성능 측정 및 비교 대조군 활용.

### Faster R-CNN (2-Stage Detector)

* 특징: 객체가 존재할 만한 후보 영역을 먼저 추출(Region Proposal Network)한 뒤, 해당 영역에 대해서만 정밀 분류와 박스 조정을 수행합니다.

* 장점: 속도는 다소 느리지만, 겹쳐진 객체나 픽셀 단위의 미세한 탐지에 있어 높은 정확도(High Accuracy)를 보장합니다.

* 사용 목적: 탐지 정밀도가 가장 우선시되는 환경에서의 성능 측정.

## 📂 03. 커스텀 모델 설계 및 최적화 (Custom Model)

Baseline 모델들의 테스트 결과를 바탕으로, 본 프로젝트의 도메인(안전 장비 미착용 탐지)에 가장 적합한 실용적인 커스텀 모델을 직접 설계하고 학습시켰습니다. 
실시간성을 고려하여 YOLO 아키텍처를 베이스로 채택하되, 다음과 같은 최적화 전략을 적용했습니다.

### 🎯 주요 개선 및 학습 전략

#### 1. 데이터 불균형(Data Imbalance) 완화

  * 학습 전 라벨 통계를 분석하여, 클래스 간 데이터 분포의 불균형을 파악했습니다.

  * 탐지율이 낮은 특정 소수 클래스(예: 안전장비 미착용)의 인식률을 높이기 위해 데이터 증강(Augmentation) 기법을 세밀하게 적용했습니다.

#### 2. 단계적 전이 학습(Transfer Learning) 및 미세 조정(Fine-tuning)
  * Phase 1 (Network Freezing): 모델의 강력한 특징 추출 능력을 잃지 않기 위해, 초기 가중치(Pre-trained Backbone)를 Freeze(동결)한 상태로 학습을 진행했습니다.
    이를 통해 모델이 공사 현장의 새로운 분류 기준을 빠르게 안정적으로 학습하도록 유도했습니다.

  * Phase 2 (Unfreeze & Fine-Tuning): 1단계에서 확보한 베스트 가중치(best.pt)를 로드한 뒤, 네트워크 잠금을 해제(freeze=0)했습니다.
    과적합(Overfitting)을 방지하기 위해 학습률(Learning Rate)을 매우 낮게 설정하고, 전체 레이어가 공사 현장 데이터에 최적화되도록 세밀하게 튜닝했습니다.

    
#### 💡 Insight
단순히 완성된 모델 라이브러리를 호출하는 것에 그치지 않고, 네트워크의 Layer 동결(Freeze) 기법을 활용한 단계별 전이 학습을 직접 제어해 보았습니다. 이 과정을 통해 한정된 데이터셋 환경에서도 과적합을 효과적으로 억제하고 탐지 성능을 극대화하는 모델 최적화 역량을 확보했습니다.

## 📂 04. 최종 모델 비교 분석 (Comparison)

학습된 각 모델의 성능을 정량적으로 평가하고 분석한 결과입니다.

### 📊 모델별 성능 요약

| 모델명    |   주요 평가지표(mAP@0.5)      |  주요 특이사항                     |
| ---------- | ---------------------------- |------------------------------------|
| Faster R-CNN|  0.7766                     |  가장 안정적인 탐지 성능 기록        |
| YOLOv8     | F1-Score(Helmet 79.8%)       | 실시간성 우수, 헬멧/사람 탐지에 강점 |
| Custom Model| 낮은 mAP                     | 탐지 원리 학습 및 구조 이해 목적 실험|



## 🔍 분석 결론

* 성능 측면: 사전 학습된 Backbone을 활용한 Faster R-CNN과 YOLOv8이 실제 현장 적용 가능성이 높은 우수한 성능을 보였습니다. 특히 Faster R-CNN은 안정적인 mAP를, YOLOv8은 우수한 실시간 탐지 능력을 입증했습니다.

* 실험적 가치: 직접 설계한 Custom Model은 사전 학습 모델 대비 탐지율은 낮았으나, Target Encoding, Multi-anchor 구조, Loss Function 설계, NMS 구현 등 객체 탐지의 핵심 로직을 밑바닥부터 직접 구현하며 AI 모델의 내부 동작 원리를 깊이 이해하는 계기가 되었습니다.
























