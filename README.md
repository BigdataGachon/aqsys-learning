# aqsys-learning

서울 중구 기상 측정소의 실측 데이터를 기반으로 향후 24시간의 미세먼지(PM10, PM2.5) 농도를 예측하는 머신러닝 파이프라인입니다.

## 프로젝트 개요

`pm_prediction.ipynb`가 메인 파일입니다. Google Colab에서 실행하며, 다음 흐름으로 구성됩니다.

```
실측 데이터 (data.csv)
  → EDA → 전처리 → 피처 엔지니어링 → 모델 비교 → LightGBM 선택 → 향후 24시간 예측
```

## 주요 파일

| 파일 | 설명 |
|------|------|
| `pm_prediction.ipynb` | **메인 노트북** — 전체 파이프라인 (Google Colab) |
| `data.csv` | 서울 중구 측정소 시간별 실측 데이터 (2025년, 8760행) |
| `result/model_pm10.pkl` | 학습된 PM10 예측 모델 (LightGBM) |
| `result/model_pm25.pkl` | 학습된 PM2.5 예측 모델 (LightGBM) |
| `result/features.pkl` | 추론 시 사용하는 피처 목록 |

> `train_model.py` / `generate_dataset.py`는 LSTM 실험용 별도 스크립트입니다.

## 데이터

- **출처:** 서울 중구 기상 측정소 (station_id: 108)
- **기간:** 2025-01-01 ~ 2025-12-31 (시간별, 8,760행)
- **원본 컬럼:** `temp`, `humidity`, `wind_speed`, `wind_dir`, `precip`, `pm10`, `pm25`, `o3`, `no2`

## 피처 엔지니어링

| 유형 | 피처 |
|------|------|
| 시간 파생 | `hour`, `month`, `weekday`, `season` |
| 기상 | `temp`, `humidity`, `wind_speed`, `wind_dir`, `precip` |
| Lag | `pm10_lag1/2/3/6/12/24`, `pm25_lag1/2/3/6/12/24` |
| Rolling 평균 | `pm10_roll3/24`, `pm25_roll3/24` |

## 모델 비교 결과

테스트 기간: 2025-12-01 ~ 2025-12-31 (마지막 30일)

| 모델 | PM10 RMSE | PM10 MAE | PM2.5 RMSE | PM2.5 MAE |
|------|-----------|----------|------------|-----------|
| LinearRegression | 5.58 | 3.82 | 4.20 | 2.85 |
| RandomForest | 5.60 | 3.75 | 4.48 | 3.01 |
| XGBoost | 5.91 | 4.10 | 4.41 | 2.92 |
| **LightGBM** | **5.24** | **3.47** | **3.83** | **2.61** |

단위: μg/m³

## 예측 방식

최종 모델은 LightGBM이며, 과거 데이터를 입력으로 받아 1시간씩 auto-regressive하게 추론해 **향후 24시간 PM10·PM2.5**를 예측합니다.

```python
from predict_pm import predict_pm

# 최소 24시간치 과거 데이터 (DataFrame) 입력
result = predict_pm(input_df)
# {'pm10': [...], 'pm25': [...]}  ← 향후 24시간 예측값
```

## 실행 환경

Google Colab에서 실행합니다. `data.csv`를 Google Drive에 업로드한 뒤 아래 경로를 맞춰주세요.

```python
FILE_PATH = '/content/drive/MyDrive/data.csv'
```

필요 패키지:
```bash
pip install lightgbm xgboost scikit-learn pandas numpy matplotlib seaborn joblib
```
