# 등록하지 않은 MP4 한 편 분석하기

측면에서 고정한 영상에서 원판과 가까운 쪽 관절을 추적한다. 준비부터 일어선 자세까지 발끝·손끝·원판 테두리 전체가 보이는 영상을 사용한다. 분석에는 원판 베이스라인 가중치, 원본 YOLO26n-pose와 ffprobe가 필요하다. Mac에서 ffprobe는 `brew install ffmpeg`로 설치할 수 있다.

## 실행 코드와 배포 모델

새 MP4의 진입점은 `wlcv.runtime.pipeline.analyze_file`이다. CLI에서도 같은 함수를 호출한다. 반환값은 결과 폴더와 요약 딕셔너리다.

```python
from wlcv.runtime.pipeline import analyze_file

folder, summary = analyze_file(
    "my lift.mp4", near_side="left", facing="left",
    manual_events={"lift_off": 11, "knee_pass": 27},
)
print(summary["manual_phase_interval"]["rise_px"])
```

| 위치 | 역할 |
| --- | --- |
| `wlcv/data/` | MP4 검사와 원본 프레임 디코딩 |
| `wlcv/runtime/inference.py`, `tracking.py` | 원판·자세 추론, 원판 선택과 시간 연결 |
| `wlcv/runtime/pose.py`, `measurements.py` | 관절 좌표 판독, 두 시점 상승량과 구간 지표 계산 |
| `wlcv/runtime/overlay.py`, `recovery.py` | 오버레이 영상과 캐치→피니시 그래프 저장 |
| `wlcv/evaluation/` | 등록 영상의 선수 분할 검사, 수동 정답과의 비교, 기존 실험의 import 경로 |

프로젝트 루트에서 아래 파일을 준비하고 `.env`에 `WL_MODEL_DIR=models`를 지정한다. 원본 체크포인트는 실험 폴더에 보존한다.

| 로컬 원본 | `models/pytorch/`에 복사할 파일명 | 사용 시점 |
| --- | --- | --- |
| `outputs/runs/plate_baseline/weights/best.pt` | `plate_baseline.pt` | 항상 |
| `models/yolo26n-pose.pt` | `yolo26n-pose.pt` | 항상 |
| `outputs/plate_cutmix_3090/20261126/runs/cutmix/weights/best.pt` | `plate_cutmix.pt` | CutMix 모드 |
| `outputs/plate_lr_grid/20261126/runs/lr0005_cosine/weights/best.pt` | `plate_cosine.pt` | 무릎 통과·접촉이 모두 지정된 경우 |

`WL_MODEL_DIR`를 지정하면 새 MP4는 이 폴더의 모델을 읽는다. 코사인 파일이 없으면 기존 추적 중심으로 상승량을 계산하고 사유를 남긴다. 변수를 비워 두면 기존 설치의 가중치 경로를 사용한다. 등록 영상 ID는 학습 선수 명단과 가중치를 같이 검사하기 위해 기존 실행 폴더를 사용한다. `WL_OUTPUT_DIR`는 새 MP4의 결과 폴더이고 기본값은 `outputs/file_analysis`다. 상대 경로는 프로젝트 루트 기준이다.

기본 PyTorch 경로는 CPU·Mac MPS에서 실행한다. NVIDIA 배포에서는 `WL_API_BACKEND=tensorrt`로 설정하고 네 모델의 FP16 엔진을 `models/tensorrt/`에서 읽는다. 엔진 준비와 합성 영상 검증은 [GPU 배포 안내](gpu_fastapi.md)를 따른다.

## 시점 없이 실행

프로젝트 루트에서 입력 파일 경로와 두 방향을 지정한다. 공백이 있는 파일 경로는 따옴표로 감싼다.

```sh
uv run --frozen --env-file .env python -m wlcv.analyze "./videos/my lift.mp4" --near-side left --facing left --device mps
```

- `--near-side`는 카메라에 가까운 **선수 자신의** 왼쪽 또는 오른쪽이다. `--facing`은 선수가 향한 **화면의** 왼쪽 또는 오른쪽이다. 각각 `left`나 `right`를 선택한다.
- 기본 원판 모델은 베이스라인이다. `--plate-mode cutmix`를 추가하면 베이스라인 출력이 비었을 때 보조 모델을 사용한다.
- 프레임별 검출·모델 출처, 원판 궤적과 판독 가능한 체간 각도를 저장한다. CSV의 `forward_px`는 어깨에 대한 원판 박스 중심의 가로 거리이며 선수 앞쪽이 양수다. 구간을 모르는 요약 지표는 비워 두고 보류 사유를 적는다.

## 아는 시점만 지정하기

수동 프레임 번호는 첫 프레임을 0번으로 센다. 이륙·무릎 통과를 주면 원판 상승량, 접촉까지 주면 앞·뒤 구간의 평균 수직속도(px/s), 앞/뒤·뒤/앞 배수와 무릎 통과 후의 판정을 표시한다. 무릎 통과나 접촉 중 하나를 ±1프레임 옮겨 판정이 바뀐 경우에만 속도와 배수 범위가 함께 표시된다. 캐치만 주면 캐치 순간의 어깨 기준 바 위치를, 캐치·피니시를 주면 두 시점 사이의 곡선을 확인할 수 있다.

```sh
uv run --frozen --env-file .env python -m wlcv.analyze data/raw/videos/c04_sn_f_01.mp4 --near-side left --facing left --device mps \
  --lift-off 11 --knee-pass 27 --drive 43 --catch 78 --finish 107
```

기본 설정에서는 실행마다 `outputs/file_analysis/<실행 시각>/` 아래에 `overlay.mp4`, `frames.csv`, `summary.json`, `timing.json`을 저장한다. 캐치·피니시 구간에 거리나 체간 관측이 있으면 `catch_to_finish.png`도 저장한다. 그래프는 원본 프레임 번호, CSV는 원본 표시 시각(ms)을 사용한다. 오버레이는 일정한 FPS로 저장한다. 촬영 FPS가 가변인 영상의 시간 측정은 CSV를 사용한다. 화면 밖으로 잘린 원판의 박스 중점은 실제 허브 위치와 다를 수 있다.

파일 입력은 같은 이름의 등록 영상이 있어도 라벨이나 선수 분할표를 참조하지 않는다. 기존 ID 입력은 등록된 수동 시점과 사람별 검증 분할 검사를 그대로 사용한다.

## Mac에서 기존 검증 영상으로 대조

c04와 a03 원본을 등록표에 없는 이름으로 열었다. 두 방향은 모두 left, 입력 640, 원판은 베이스라인, 자세는 원본 YOLO26n-pose를 사용했다. 검증용 수동 박스는 추론 완료 후 IoU 0.50 기준으로 채점했다.

| 파일과 모드 | 디코딩 | 수동 박스와 일치 | 수동 시점을 사용한 결과 |
| --- | ---: | ---: | --- |
| c04 단일 · 시점 없음 | 116장 | 69/69 | 이륙·캐치 구간 측정 보류 |
| c04 단일 · 시점 5개 | 116장 | 69/69 | 상승량 +160.514 px, 최대 후방 간격 92번 82.348 px, 캐치→피니시 체간 −21.847° |
| a03 단일 · 시점 없음 | 122장 | 27/40 | 보조 모델 없음 |
| a03 CutMix · 시점 없음 | 122장 | 35/40 | 4~6번 및 10~14번 추가 정탐, 기존 정탐 손실 0장 |

각 파일 모드의 프레임 번호·원본 시각·검출 및 자세 출력은 같은 모드로 저장했던 ID 분석과 일치했다. c04의 기존 ID 명령도 다시 실행했다. 다섯 오버레이를 끝까지 읽어 원본과 장수·표시 해상도가 같은지 확인했다. 이 비교는 입력 경로를 바꾼 뒤의 재현성 검증이다.

## 무릎 통과부터 접촉까지의 원판 상승량

기본 모드에서는 Mac 베이스라인이 원판 궤적을 만든다. `--plate-mode cutmix`를 선택하면 베이스라인이 놓친 프레임을 CutMix로 보완한다. 어느 모드든 무릎 통과와 접촉 시점이 모두 있으면 그 두 장에서만 `lr0=0.0005` 코사인 가중치를 추가로 실행한다. 이륙 시점은 이 측정에 필요하지 않다.

```sh
# 검증 영상 ID는 저장된 시점을 사용한다. 기본은 Mac 베이스라인 단독 추적이다.
.venv/bin/python -m wlcv.analyze c04_sn_f_01

# 파일 입력은 사용자가 0부터 센 프레임 번호를 준다. 이 예시는 가림 복구 옵션을 켠다.
.venv/bin/python -m wlcv.analyze data/raw/videos/c04_sn_f_01.mp4 \
  --near-side left --facing left --knee-pass 27 --drive 43 --plate-mode cutmix
```

원본을 읽는 중 오버레이를 그리기 전에 해당 두 장을 보관한다. 코사인은 각 장을 입력 크기 640으로 독립 추론한다. 두 장 모두 직접 검출 신뢰도 0.10 이상이고 코사인 박스와 기존 추적 박스의 IoU가 각각 0.50 이상이면 코사인 박스 중심으로 상승량을 계산한다. 한 장이라도 통과하지 못하면 **두 시점 모두** 기존 추적 좌표를 사용한다. 기존 추적 좌표까지 빠진 시점이 있으면 측정을 보류한다.

`summary.json`의 `knee_to_contact_rise`에 위쪽을 양수로 한 `rise_px`, 두 프레임 번호, 출처(`source`)와 시점별 판정(`endpoints`)이 들어 있다. 출처는 `cosine`, `tracking`, 측정 보류 시 `null`이다. 코칭 결과의 `pulls.late`는 이 상승량을 원본 경과 시간(초)으로 나눈 값이다. `pulls.early`는 이륙·무릎 통과의 추적 좌표에서 계산하며, `pulls.ratio`는 `early / late`다. 후반 상승량이 1px 이하면 비율을 보류한다. 오버레이와 프레임 CSV에는 영상 전체의 추적 좌표를 기록한다.

기존 실험 경로를 사용할 때 코사인 가중치는 `outputs/plate_lr_grid/20261126/runs/lr0005_cosine/weights/best.pt`다. 다른 위치에 저장했다면 환경변수 `WL_COSINE_WEIGHTS`에 경로를 지정한다. 가중치 파일이 없으면 기존 추적 좌표로 계산하고 사유를 남긴다. 무릎 통과 또는 접촉을 지정하지 않은 파일에서는 코사인 가중치를 열지 않는다.

### 검증 6편에서 재현한 결과

Mac CPU에서 각 검증 영상 ID를 `--plate-mode cutmix`로 끝까지 분석했다. 687프레임의 박스·중심을 포함하는 궤적을 저장했고, 오버레이도 687프레임까지 읽었다. 동작 구간 478프레임 중 수동 박스가 있는 270장에서 IoU 0.50으로 채점했다. 정탐 265장, 미탐 5장, 오탐 0장이고 최장 연속 좌표 누락은 a03의 3프레임이었다.

| 선수 | 무릎 통과 → 접촉 | 상승량 출처 | 수동 상승량(px) | 분석 상승량(px) | 오차(px) |
| --- | --- | --- | ---: | ---: | ---: |
| a03 | 19 → 25 | 기존 추적 | 71 | 70.33 | −0.67 |
| a05 | 18 → 24 | 코사인 | 114 | 115.94 | +1.94 |
| b02 | 30 → 36 | 코사인 | 326 | 324.29 | −1.71 |
| b03 | 24 → 31 | 코사인 | 203 | 199.21 | −3.79 |
| c04 | 27 → 43 | 코사인 | 218 | 215.43 | −2.57 |
| c08 | 24 → 33 | 코사인 | 298 | 295.29 | −2.71 |

선수 6명의 상승량 평균 절대오차는 **2.23 px**였다. a03은 접촉 25번에서 코사인 직접 검출을 보류해 이 선수의 상승량만 기존 좌표로 계산했다. 같은 검증 영상에서 [76번 실험](plate_cosine_fallback.md)의 결과를 실제 분석 명령으로 재현한 수치다.
