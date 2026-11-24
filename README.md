# 스내치 측면 영상의 원판 탐지와 자세 추정

카메라에 가까운 원판을 표시하고 YOLO26n이 같은 프레임에서 찾은 위치와 비교합니다. 학습용 라벨은 8명의 45프레임, 검증용 라벨은 다른 6명의 39프레임입니다. 학습에는 원판이 없는 6프레임이 포함됩니다. 검증에는 원판이 있는 프레임만 있습니다. 자세 추정은 같은 사람별 분할의 검증 영상에서 사전학습 YOLO26n-pose의 관절 위치를 확인합니다.

## 로컬 환경 준비

프로젝트 루트에서 실행합니다. Python 3.12와 uv를 사용합니다.

```sh
uv sync --locked
cp -n .env.example .env
uv run --env-file .env jupyter lab
```

`.env`에서 원본 영상(`WL_VIDEO_DIR`), 시작 가중치(`WL_WEIGHTS`), 추출 데이터(`WL_DATASET_DIR`), 학습 결과(`WL_RUNS_DIR`)를 지정합니다. 상대 경로는 프로젝트 루트 기준입니다. 학습 장치 `WL_TRAIN_DEVICE`는 Mac에서 `mps`, CUDA 첫 GPU에서 `0`, CPU에서 `cpu`입니다. 환경을 변경하면 JupyterLab을 다시 시작합니다.

VS Code에서 사진은 보이는데 클릭해도 점이 생기지 않고 Jupyter 출력에 `ipyevents` 스크립트를 찾을 수 없다고 표시되면, 프로젝트 루트에서 가상환경에 들어 있는 스크립트를 복사합니다.

```sh
mkdir -p .venv/share/jupyter/nbextensions/ipyevents
cp -R .venv/lib/python3.12/site-packages/ipyevents/nbextension/static/. .venv/share/jupyter/nbextensions/ipyevents/
```

노트북에서 **커널 다시 시작** 후 **모든 셀 실행**을 누르면 새 판독 화면을 사용할 수 있습니다.

새 컴퓨터에서는 원본 MP4를 `data/raw/videos/`에 두고 YOLO26n 사전학습 가중치를 준비합니다. 예시 환경 파일의 가중치 경로를 사용할 때의 명령입니다.

```sh
mkdir -p models
curl --fail --location --output models/yolo26n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
```

## Mac에서 측면 영상 한 편 분석하기

로컬 환경 준비 후 프로젝트 루트에서 실행합니다. c04 원본 영상, 원판 베이스라인 `outputs/runs/plate_baseline/weights/best.pt`와 같은 실행의 `train_people.csv`, 원본 자세 모델 `models/yolo26n-pose.pt`가 필요합니다.

```sh
uv run --frozen --env-file .env python -m wlcv.analyze c04_sn_f_01
```

검증 영상을 처음부터 끝까지 Mac CPU에서 읽습니다. 터미널에 원판 상승량, 이륙 11번→무릎 통과 27번과 무릎 통과→접촉 43번의 평균 수직 속도, 캐치 78번의 바−어깨 위치가 표시됩니다. 네 시점은 트레이너가 저장한 수동 라벨입니다. 프레임 번호는 0부터 셉니다.

| 로컬 결과 | 확인할 내용 |
| --- | --- |
| `overlay.mp4` | 초록은 직접 검출, 주황은 시간 연결입니다. 미탐에서는 궤적이 끊깁니다. 하늘색은 선수 기준 가까운 쪽의 자세 예측입니다. 수행자 후보가 한 명일 때 신뢰도 0.50 이상인 관절만 표시합니다. |
| `frames.csv` | 원본 시각(ms), 수동 시점, 원판 판정과 원래 신뢰도, 박스 중심 x·y(px), 가까운 어깨 x·y(px), 사람 후보 수와 표시한 관절 수가 한 프레임당 한 줄입니다. |
| `summary.json` | 전체 판정별 프레임 수, 두 시점의 y좌표, 상승량과 구간 내 관측 수입니다. 상승량은 원본 해상도에서 이륙 y − 무릎 통과 y입니다. 양 끝 중 미탐이 있으면 해당 프레임을 기록하고 상승량은 비워 둡니다. |

JSON의 `coaching.pulls`에는 앞·뒤 구간의 속도(px/s), 앞/뒤 비율과 끝점 검출 상태가 있습니다. `coaching.catch`에서 양수는 선수가 향한 쪽과 위쪽입니다. 모델 추정치이며 측정에 필요한 점이 없으면 해당 값과 보류 사유를 확인할 수 있습니다. [실험별 결과와 문헌 근거](docs/experiment-evidence.md)

직접 검출은 신뢰도 0.10 이상인 최상위 원판 후보입니다. 그 아래에서는 마지막 직접 검출 후 최대 2프레임 동안 예상 중심으로부터 직전 박스 너비의 0.5배 이내에 있는 후보(신뢰도 0.001 이상)를 연결합니다. 미탐은 이 규칙이 채택한 후보가 없다는 뜻입니다. 정탐 여부는 정답과 대조해야 알 수 있습니다.

결과는 Git에서 제외한 `outputs/video_analysis/c04_sn_f_01/<실행 시각>/`에 저장합니다. 오버레이는 원본과 1대1로 대응하는 무음 영상이며 원본 프레임 간격 중앙값에 해당하는 30 fps로 재생합니다. 세부 시간 분석에는 CSV의 원본 시각을 사용합니다.

## 작업별 노트북

| 작업 | 노트북 |
| --- | --- |
| 프레임에서 원판 박스를 그리고 저장본 다시 열기 | [원판 라벨링](notebooks/annotation/01_plate_boxes.ipynb) |
| 동작 이벤트의 프레임 번호 기록 | [동작 시점 판독](notebooks/annotation/02_phase_events.ipynb) |
| 가까운 쪽 관절을 직접 찍거나 가림·불확실로 기록 | [수동 관절 판독](notebooks/annotation/03_pose_points.ipynb) |
| 같은 프레임을 이전 점 없이 다시 판독하고 두 회차의 차이 확인 | [관절 2차 판독](notebooks/annotation/04_pose_repeatability.ipynb) |
| b04 9·21번과 a01 5번의 가까운 골반을 두 차례 직접 찍기 | [학습 영상 골반 수동 검수](notebooks/annotation/05_train_hip_review.ipynb) |
| 학습 후보 8장의 수행자 전신 박스와 가까운 네 관절을 표시 | [학습용 수행자·관절 판독](notebooks/annotation/06_train_pose_candidates.ipynb) |
| a08 무릎 통과·drive에서 오른쪽 무릎과 발목을 나란히 판독 | [오른쪽 무릎·발목 판독](notebooks/annotation/07_a08_right_knee_ankle.ipynb) |
| 무릎·발목 직접점이 있는 학습 1장과 검증 2장에 수행자 전신 박스 표시 | [수행자 박스 판독](notebooks/annotation/08_pose_performer_boxes.ipynb) |
| 학습·검증 추가 후보에서 가까운 무릎·발목과 수행자 박스를 직접 표시 | [무릎·발목 추가 판독](notebooks/annotation/09_knee_ankle_candidates.ipynb) |
| 머리 위에서 팔이 가려진 두 장의 수행자 박스 재판독과 동일 예측 IoU 비교 | [보이는 수행자 박스 다시 그리기](notebooks/annotation/10_visible_performer_boxes.ipynb) |
| a05 30·41·60번의 오른쪽 무릎·발목 수동점과 YOLO의 좌우 예측 거리 비교 | [a05 좌우 관절 판독](notebooks/annotation/11_a05_pose_side_review.ipynb) |
| a05 84·87번 양쪽 무릎을 직접 판독하고 같은 쪽·교차 연결 거리 비교 | [양쪽 무릎 좌우 대응](notebooks/annotation/12_a05_bilateral_knees.ipynb) |
| 한 프레임의 정답·예측 비교, 기본값 학습, FiftyOne에서 검증셋 검수 | [YOLO26n 실험](notebooks/experiments/01_plate_baseline.ipynb) |
| 학습에 쓰지 않은 사람의 원판 없는 영상 전체에서 오탐 세기 | [원판 없는 영상의 오탐 평가](notebooks/experiments/02_false_positives.ipynb) |
| 사람별 데이터 분포, 학습 곡선, 신뢰도 비교, 중심 오차와 가림 실험 | [원판 탐지 보고서](notebooks/experiments/03_plate_report.ipynb) |
| 원판 가림 추가 학습 예시와 동일 검증셋 비교 | [원판 가림 증강 실험](notebooks/experiments/04_plate_occlusion.ipynb) |
| 회색과 학습 배경 질감 가림을 비교하고 새 패턴에서 검증 | [가림 질감 비교](notebooks/experiments/05_plate_texture_occlusion.ipynb) |
| 준비부터 받기까지 신뢰도별 연속 미탐과 랙 오탐 확인 | [원판 연속 미탐 평가](notebooks/experiments/06_plate_temporal_misses.ipynb) |
| 베이스라인·회색·질감 모델이 같은 구간에서 놓친 프레임 비교 | [가림 모델 연속 미탐 비교](notebooks/experiments/07_plate_temporal_model_comparison.ipynb) |
| 질감 가림 위치별 원판 검출과 빈 바닥 오탐 비교 | [질감 패치 위치 실험](notebooks/experiments/08_plate_patch_position.ipynb) |
| 아래 윤곽 가림에서 같은 예측을 합성 전 박스와 보이는 부분 박스로 평가 | [가림 후 정답 박스 비교](notebooks/experiments/09_plate_visible_box_evaluation.ipynb) |
| 검증 31장에서 같은 질감으로 중앙·아래쪽 가림 비율을 높이며 두 모델의 신뢰도와 후보 소실 비교 | [가림 비율별 원판 탐지](notebooks/experiments/10_plate_occlusion_dose.ipynb) |
| 두 프레임의 원판 하단 안쪽·바깥·경계에서 베이스라인 비교 | [원판 하단 경계 가림](notebooks/experiments/11_plate_rim_boundary.ipynb) |
| 실제 77프레임의 수동 원판 중심과 프레임별·시간 연결 결과 비교 | [원판 중심 궤적](notebooks/experiments/12_plate_center_tracking.ipynb) |
| 같은 저신뢰 후보를 직전 박스 너비의 0.5배·1.0배 거리에서 연결 | [허용 거리 비교](notebooks/experiments/13_plate_link_distance.ipynb) |
| 다른 검증 영상 3편의 171프레임에서 프레임별 탐지와 0.5배 시간 연결 비교 | [다른 검증 영상의 추적 비교](notebooks/experiments/14_additional_plate_tracking.ipynb) |
| b02·c07 원본 영상 4편 전체에서 저신뢰 원판의 복구와 연속 미탐 확인 | [전체 영상의 저신뢰 연결](notebooks/experiments/15_low_confidence_tracking.ipynb) |
| 고신뢰 박스의 이력 갱신 여부만 바꾸고 같은 417프레임에서 복구·오연결 확인 | [고신뢰 박스의 궤적 반영](notebooks/experiments/16_high_history_update.ipynb) |
| 원판 크기와 밝기 증강을 하나씩 끄고 검증 결과 비교 | [scale·hsv_v 통제실험](notebooks/experiments/17_scale_hsv_ablation.ipynb) |
| 검증 영상에서 어깨·골반·무릎·발목 예측을 원본과 비교 | [YOLO26n-pose 자세 추정 베이스라인](notebooks/experiments/18_pose_baseline.ipynb) |
| 다른 검증 선수 3명의 무릎 통과에서 가까운 복사뼈의 세로 차이 확인 | [무릎 통과 발목 비교](notebooks/experiments/19_transition_ankles.ipynb) |
| 같은 세 프레임에서 모델 발목점과 복사뼈·앞쪽 접힘점의 거리 비교 | [발목 접힘점 판독](notebooks/experiments/20_ankle_fold_landmark.ipynb) |
| 이륙·무릎 통과에서 가까운 어깨와 골반을 찍고 체간 기울기 변화와 옷 가림 추정 구분 | [두 시점의 체간 기울기](notebooks/experiments/21_trunk_inclination.ipynb) |
| 같은 12프레임의 어깨·골반을 무작위 순서로 다시 찍고 두 회차의 체간 변화 비교 | [체간 기울기 반복 판독](notebooks/experiments/22_trunk_repeatability.ipynb) |
| 12프레임의 YOLO26n-pose 체간 각도와 두 수동 판독의 변화 방향·크기 비교 | [모델과 수동 체간 기울기 비교](notebooks/experiments/23_pose_trunk_model.ipynb) |
| a05·b02에서 어깨 또는 골반만 바꿨을 때 체간 변화 방향 비교 | [체간 관절 한 점 교체](notebooks/experiments/24_trunk_joint_swap.ipynb) |
| 실제 학습 영상에서 가까운 골반을 짚을 수 있는 장면 수와 두 번의 추정점 차이 | [학습 영상 골반 판독](notebooks/experiments/25_train_hip_visibility.ipynb) |
| 같은 세 프레임에서 트레이너의 두 골반 추정점과 예비 판독을 비교 | [학습 영상 골반 반복 비교](notebooks/experiments/26_train_hip_repeat.ipynb) |
| b04 두 시점에서 눈에 보이는 고관절 접힘점·소켓 추정점·YOLO 골반을 비교 | [고관절 접힘점과 체간 기울기](notebooks/experiments/27_hip_fold_comparison.ipynb) |
| b04 9·21번의 왼쪽 견봉을 찍고 소켓 중점과 연결해 YOLO 체간 변화와 비교 | [왼쪽 견봉과 체간 기울기](notebooks/experiments/28_b04_acromion_trunk.ipynb) |
| 세 수행자 박스·무릎·발목을 YOLO 라벨로 변환하고 Mac에서 1 epoch 학습 | [자세 추정 시험 학습](notebooks/experiments/29_pose_smoke.ipynb) |
| 자세 학습 5장·검증 5장으로 첫 파인튜닝, 수행자 미탐과 다리 관절 오차 전후 비교 (a05 별도) | [무릎·발목 첫 파인튜닝](notebooks/experiments/30_pose_first_finetune.ipynb) |
| b02에서 시간 연결 전후 원판 궤적, 35번 복구와 97·98번 실패를 나란히 비교 | [시간 연결 복구와 실패](notebooks/experiments/31_b02_link_replay.ipynb) |

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
| `wlcv/annotation/` | 박스·시점·관절 판독 화면. 개인 기록은 `data/annotations/` 아래 `personal_plate_boxes/`, `personal_point_labels.csv`, `personal_pose_points/`에 저장 |
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
