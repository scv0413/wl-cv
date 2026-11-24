# 스내치 측면 영상의 원판 탐지

카메라에 가까운 원판을 표시하고 YOLO26n이 같은 프레임에서 찾은 위치와 비교합니다. 학습용 라벨은 8명의 45프레임, 검증용 라벨은 다른 6명의 39프레임입니다. 학습에는 원판이 없는 6프레임이 포함됩니다. 검증에는 원판이 있는 프레임만 있습니다.

## 로컬 환경 준비

프로젝트 루트에서 실행합니다. Python 3.12와 uv를 사용합니다.

```sh
uv sync --locked
cp -n .env.example .env
uv run --env-file .env jupyter lab
```

`.env`에서 원본 영상(`WL_VIDEO_DIR`), 시작 가중치(`WL_WEIGHTS`), 추출 데이터(`WL_DATASET_DIR`), 학습 결과(`WL_RUNS_DIR`)를 지정합니다. 상대 경로는 프로젝트 루트 기준입니다. 학습 장치 `WL_TRAIN_DEVICE`는 Mac에서 `mps`, CUDA 첫 GPU에서 `0`, CPU에서 `cpu`입니다. 환경을 변경하면 JupyterLab을 다시 시작합니다.

새 컴퓨터에서는 원본 MP4를 `data/raw/videos/`에 두고 YOLO26n 사전학습 가중치를 준비합니다. 예시 환경 파일의 가중치 경로를 사용할 때의 명령입니다.

```sh
mkdir -p models
curl --fail --location --output models/yolo26n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
```

## 작업별 노트북

| 작업 | 노트북 |
| --- | --- |
| 프레임에서 원판 박스를 그리고 저장본 다시 열기 | [원판 라벨링](notebooks/annotation/01_plate_boxes.ipynb) |
| 동작 이벤트의 프레임 번호 기록 | [동작 시점 판독](notebooks/annotation/02_phase_events.ipynb) |
| 한 프레임의 정답·예측 비교, 기본값 학습, FiftyOne에서 검증셋 검수 | [YOLO26n 실험](notebooks/experiments/01_plate_baseline.ipynb) |
| 학습에 쓰지 않은 사람의 원판 없는 영상 전체에서 오탐 세기 | [원판 없는 영상의 오탐 평가](notebooks/experiments/02_false_positives.ipynb) |
| 사람별 데이터 분포, 학습 곡선, 신뢰도 비교, 중심 오차와 가림 실험 | [원판 탐지 보고서](notebooks/experiments/03_plate_report.ipynb) |

### 학습 결과를 보고서 그림으로 만들기

완료한 실행의 `results.csv`와 `weights/best.pt`를 사용하는 원판 탐지 보고서 노트북을 실행하세요. 새 그림은 해당 실행의 `report/`에 PNG로 저장됩니다. 기준값을 바꿔 보고 싶으면 노트북 상단의 `thresholds`를 수정하세요. 원판과 랙을 가리는 XAI 셀에서는 마스크 위치와 대상 프레임도 확인할 수 있습니다.

### 학습 전 영상 입력 확인

YOLO26n 실험 노트북의 첫 코드 셀에서 라벨이 있는 영상의 확장자, 헤더 브랜드, 코덱과 마지막 라벨 프레임을 확인합니다. 파일 형식은 FFmpeg의 `ffprobe`로 읽습니다. Mac에서 `ffprobe`가 없으면 `brew install ffmpeg`로 설치하세요.

노트북에서 `프레임 읽기`가 실패하거나 미검사인 행의 `메모`를 확인한 뒤 학습을 시작하세요.

### FiftyOne으로 원판 검수하기

위 YOLO26n 실험 노트북을 순서대로 실행하세요. 처음에는 `run_training = False`로 두고 프레임을 확인합니다. 마지막 셀에서 `open_review = True`로 바꾸면 로컬 FiftyOne에서 검증셋의 정답 박스를 볼 수 있습니다. FiftyOne은 `uv sync --locked`로 설치되는 개발 의존성에 포함되어 있습니다.

학습 후에는 확인할 실행 폴더의 `weights/best.pt`를 불러온 뒤 검수 셀을 다시 실행하세요. `ground_truth`와 `predictions`를 함께 켜고, `eval_fp`와 `eval_fn`으로 오탐·미탐 프레임을 찾습니다. 화면의 수치는 신뢰도 0.25, IoU 0.5에서 센 박스 개수입니다. 커널을 다시 시작하면 로컬 파일에서 검수 화면을 다시 만들 수 있습니다.

## 코드와 데이터 위치

| 디렉토리 | 역할 |
| --- | --- |
| `data/raw/videos/` | 로컬 원본 MP4 |
| `data/annotations/plate_boxes/` | 베이스라인에 쓰는 84개 프레임의 원판 박스. 빈 파일은 검수한 배경, 파일이 없는 프레임은 미라벨 |
| `data/annotations/reference_plate_boxes/` | 초기 검수용 박스 40개. 학습용 라벨과 영상·프레임 조합이 모두 다름 |
| `data/annotations/`의 CSV | `point_labels.csv`는 ready·catch 등 동작의 프레임 번호, `plate_frame_ranges.csv`는 원판이 보이는 범위의 예외. `reference_*.csv`는 초기 판독 자료 |
| `data/metadata/`, `data/splits/` | 영상 ID·촬영 시간·음성 평가 목록, 사람별 분할 |
| `wlcv/data/` | 영상을 순서대로 읽고 라벨이 있는 프레임을 YOLO 데이터셋으로 추출 |
| `wlcv/annotation/` | 박스와 동작 시점의 판독 화면. 수정본은 `data/annotations/personal_plate_boxes/`와 `data/annotations/personal_point_labels.csv`에 저장 |
| `wlcv/training/` | YOLO26n 학습과 실제 학습에 쓰인 사람의 기록 |
| `wlcv/evaluation/` | 같은 프레임의 정답·예측 비교, 학습 곡선, 음성 영상 오탐 평가 |
| `tests/` | 데이터 분할·좌표, 라벨 저장, 노트북 공유 형식 검증 |
| `outputs/` | 실행 시 생성되는 데이터셋과 실행별 학습 결과 |
| `models/` | 예시 환경에서 사용하는 로컬 시작 가중치 위치 |

## 터미널에서 학습

```sh
uv run --env-file .env python -m wlcv.training.train
```

첫 학습은 기존 사람별 분할과 YOLO26n 사전학습 가중치를 사용합니다. 학습 하이퍼파라미터는 설치된 Ultralytics의 기본값을 따릅니다. 실제 설정은 실행 폴더의 `args.yaml`에서, 자동 선택된 옵티마이저와 초기 학습률은 학습 시작 로그에서 확인하세요. 결과 폴더에는 지표(`results.csv`), 가중치(`weights/best.pt`), 실제 학습 인물(`train_people.csv`)이 저장됩니다. 실험 기록과 공유 전 확인 순서는 [SKILLS.md](SKILLS.md)에 있습니다.
