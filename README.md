# dcfnet-ds

**DCFNet-DS: Degree-Separated Cross Fusion for Order-Aware Explainable Intrusion Detection**

SHIELD-Net 후속 연구. 교차 네트워크의 잔차 연결을 제거해 상호작용 차수를
분리하고, 학습 가능한 차수 게이트 `alpha_k` 로 "탐지에 몇 차 상호작용이
필요한가"를 정량화한다.

---

## 프로젝트 구조

```
dcfnet-ds/
├── data/
│   ├── source/                     원본 대용량 CSV (선택, git 제외)
│   │   ├── CICIoT2023.csv
│   │   └── NF-UNSW-NB15-v3.csv
│   ├── raw/                        클래스당 <=10k 샘플링 결과
│   │   ├── CICIoT2023.csv
│   │   └── NF-UNSW-NB15-v3.csv
│   └── processed/                  자동 생성
│       ├── ciciot2023.npz / ciciot2023_meta.json
│       └── nfunsw.npz    / nfunsw_meta.json
├── src/
│   ├── __init__.py
│   ├── config.py                   경로·상수·실험 변형 정의
│   ├── check_env.py                환경 점검 (최초 1회)
│   ├── prepare_data.py             source -> raw 통제 샘플링
│   ├── preprocess.py               raw -> processed (분할 후 스케일링)
│   ├── model.py                    DCFNet-DS 모델
│   ├── train.py                    변형별 학습
│   ├── evaluate.py                 지표·혼동행렬·ROC/PR
│   └── degree_analysis.py          차수 게이트 분석 (핵심 기여)
├── outputs/
│   ├── models/  figures/  tables/
├── requirements.txt
└── run_all.py
```

## 실험 변형

| ID | 구성 | 목적 |
|----|------|------|
| m0 | 잔차 교차 x3 (SHIELD-Net) | 선행 baseline |
| m1 | 차수 분리 + 균등 가중 | 분리 자체의 효과 |
| m2 | 차수 분리 + 전역 게이트 | 차수 순수성 완전 보존 |
| m3 | **차수 분리 + 인스턴스 게이트** | **제안 모델** |

## 실행 순서

```bash
# 0) 전용 conda 환경 (기존 pytorch 환경 재사용 금지)
conda create -n dcfnet-ds python=3.11 -y
conda activate dcfnet-ds
pip install -r requirements.txt

# 0-1) 환경 점검 — 가장 먼저 실행
python -m src.check_env

# 1) 데이터 배치
#    이미 샘플링된 CSV 가 있으면 data/raw/ 에 아래 이름으로 둔다
#      data/raw/CICIoT2023.csv
#      data/raw/NF-UNSW-NB15-v3.csv
#    원본에서 시작한다면 data/source/ 에 두고
python -m src.prepare_data --dataset ciciot2023
python -m src.prepare_data --dataset nfunsw

# 2) 전처리 (train/test 분리 후 train 에만 스케일러 fit)
python -m src.preprocess --dataset all

# 3) 학습
python -m src.train --dataset ciciot2023 --variant m0
python -m src.train --dataset ciciot2023 --variant m3
#    ablation 전체
python -m src.train --dataset ciciot2023 --variant all

# 4) 평가
python -m src.evaluate --dataset ciciot2023

# 5) 차수 분석 (논문의 핵심 표·그림)
python -m src.degree_analysis --dataset ciciot2023 --variant m3

# 일괄 실행
python run_all.py --dataset ciciot2023
```

**PyCharm 실행 구성**: Run > Edit Configurations > Add(Python) >
Module name 에 `src.train`, Parameters 에 `--dataset ciciot2023 --variant m3`,
Working directory 를 프로젝트 루트로 지정한다.
Script path 가 아니라 **Module name** 을 써야 `from src import ...` 가 동작한다.

## 산출물

| 파일 | 내용 |
|------|------|
| `outputs/tables/metrics_{ds}.csv` | 변형별 Accuracy/Precision/Recall/F1/MCC |
| `outputs/tables/degree_alpha_{tag}.csv` | **공격 유형 x 차수 게이트 표** |
| `outputs/tables/purity_{tag}.csv` | 동차성 검정 (차수 분리 증명) |
| `outputs/figures/degree_alpha_{tag}.png` | 차수 사용 비율 누적 막대 |
| `outputs/figures/roc_{ds}.png` | ROC (축 범위와 라벨 일치) |

## 선행 연구 대비 변경점

1. **train/test 분리 후 스케일러 fit** — 선행 코드는 전체 데이터로 fit 했다.
2. **학습에 test 미사용** — 선행 코드는 `fit(x, y, validation_data=(x, y))` 였다.
3. **NF-UNSW-NB15-v3 식별자 제거** — IP·포트가 SHAP 상위권을 차지해
   설명이 호스트 암기를 반영하던 문제를 차단했다. 51 -> 47 피처.
4. **Adam + Focal Loss 실제 적용** — 선행 코드는 재컴파일로 RMSprop + CE 가
   실행되고 있었다.
5. **ROC/PR 축 라벨 정정** — 축 범위와 눈금 라벨을 일치시켰다.

## 검증 완료 사항

`python -m src.preprocess --dataset all` 실행 결과:

```
ciciot2023  train=(41829, 40)  test=(17928, 40)   상수 6개 제거
nfunsw      train=(39346, 47)  test=(16863, 47)   식별자 7개 제거
```

라벨 매핑이 선행 논문과 정확히 일치함을 확인했다.

## Windows GPU 참고

TensorFlow 2.10 이 네이티브 Windows GPU 를 지원한 마지막 버전이다.
2.11 이상은 Windows 에서 CPU 로만 동작하며, GPU 가 필요하면 WSL2 를 써야 한다.

이 프로젝트의 데이터 규모(약 4만 x 40)에서는 CPU 로 충분하다.
GPU 를 쓰려면 아래처럼 구성하고 `src/config.py` 의 `MODEL_EXT` 를 `".h5"` 로 바꾼다.

```bash
conda create -n dcfnet-ds-gpu python=3.10 -y
conda activate dcfnet-ds-gpu
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
pip install "tensorflow<2.11" "numpy<2.0" pandas scikit-learn matplotlib seaborn shap
```
