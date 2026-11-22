# 스내치 측면 영상의 원판 탐지

카메라에 가까운 원판을 표시하고 YOLO26n이 같은 프레임에서 찾은 위치와 비교합니다. 학습용 라벨은 8명의 45프레임, 검증용 라벨은 다른 6명의 39프레임입니다. 학습에는 원판이 없는 6프레임이 포함됩니다. 검증에는 원판이 있는 프레임만 있습니다.

## 실행

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
| 한 프레임의 정답과 사전학습 예측 비교, 학습, 지표와 예측 다시 보기 | [YOLO26n 실험](notebooks/experiments/01_plate_baseline.ipynb) |
| 학습에 쓰지 않은 사람의 원판 없는 영상 전체에서 오탐 세기 | [음성 영상 평가](notebooks/experiments/02_negative_videos.ipynb) |

원판 라벨이 없는 프레임과 사람이 검수해 빈 파일로 저장한 프레임을 구분합니다. 개인 박스 수정은 `data/annotations/personal_plate_boxes/`에, 동작 시점 수정은 `personal_point_labels.csv`에 저장합니다. 첫 학습은 보존된 `plate_boxes/`와 `person_split.csv`를 사용합니다. 전 구간 음성으로 확인한 새 영상은 `data/metadata/negative_eval.csv`에 등록합니다. 현재 등록된 음성 영상은 0편입니다.

## 코드와 데이터 위치

| 디렉토리 | 역할 |
| --- | --- |
| `data/raw/videos/` | 로컬 원본 MP4 |
| `data/annotations/` | 원판 박스, 동작 이벤트, 초기 검수 기록과 개인 라벨 |
| `data/metadata/`, `data/splits/` | 영상 ID·촬영 시간·음성 평가 목록, 사람별 분할 |
| `wlcv/data/` | 영상을 순서대로 읽고 라벨이 있는 프레임을 YOLO 데이터셋으로 추출 |
| `wlcv/annotation/` | 원판 박스 드래그, 이벤트 판독, 개인 파일 저장·재열기 |
| `wlcv/training/` | YOLO26n 학습과 실제 학습에 쓰인 사람의 기록 |
| `wlcv/evaluation/` | 같은 프레임의 정답·예측 비교, 학습 곡선, 음성 영상 오탐 평가 |
| `notebooks/` | 라벨링 화면과 단계별 실험 |
| `tests/` | 데이터 분할·좌표, 라벨 저장, 노트북 공유 형식 검증 |
| `outputs/` | 실행 시 생성되는 데이터셋과 실행별 학습 결과 |
| `models/` | 예시 환경에서 사용하는 로컬 시작 가중치 위치 |

## 터미널에서 학습

```sh
uv run --env-file .env python -m wlcv.training.train
```

기본값은 20 epoch, 입력 640, batch 4, AdamW, 초기 학습률 0.001, seed 42입니다. 결과 폴더에는 지표(`results.csv`), 가중치(`weights/best.pt`), 실제 학습 인물(`train_people.csv`)이 저장됩니다. 원본 영상·이미지·가중치와 노트북 출력은 로컬에서 확인합니다.
