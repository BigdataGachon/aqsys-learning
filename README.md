# Air Quality Prediction using LSTM

이 프로젝트는 시계열 데이터(LSTM)를 활용하여 24시간의 과거 데이터를 기반으로 다음 시간의 미세먼지(PM10) 및 초미세먼지(PM25) 농도를 예측하는 딥러닝 모델입니다.

## 🚀 프로젝트 개요

- **모델 아키텍처**: LSTM (Sequence-to-Point)
- **입력 데이터**: 과거 24시간의 기상 정보 및 오염 물질 농도 (24 timestep × 16 features)
- **출력 데이터**: 현재 시점(t+1)의 PM10, PM25 예측값
- **주요 특징**: Apple Silicon (MPS), NVIDIA GPU (CUDA) 가속 지원, Sliding Window 방식의 시계열 학습

## 📁 주요 파일 설명

- `train_model.py`: LSTM 모델 학습 및 평가 메인 스크립트
- `generate_dataset.py`: 학습용 데이터셋 생성 도구
- `air_quality_train.ipynb`: 데이터 분석 및 시각화용 Jupyter Notebook
- `air_quality_train_sample.json`: 데이터셋 샘플 구조 정보
- `.gitignore`: 대용량 데이터 및 환경 변수 제외 설정


## 📈 모델 상세 구조 (Architecture)

```text
LSTM(16 inputs → 128 hidden, 2 layers)
  ↓
LayerNorm
  ↓
Dropout(0.2)
  ↓
Linear(128 → 64) → ReLU
  ↓
Linear(64 → 2) → ReLU (Final PM10, PM25 Prediction)
```

