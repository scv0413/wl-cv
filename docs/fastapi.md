# Mac에서 스내치 분석 API 실행하기

측면 MP4 한 편을 제출하면 분석이 끝날 때까지 기다렸다가 원판 추적 요약과 결과 링크를 받습니다. 영상 전송, 추론, 결과 저장은 같은 Mac에서 실행합니다.

## 서버 시작

프로젝트 루트에서 실행합니다. Python 3.12, uv, ffprobe와 배포용 원판·자세 모델이 필요합니다. [모델 위치와 MP4 준비 방법](analyze_mp4.md)을 참고하세요.

```bash
uv sync --locked --no-dev --inexact
cp -n .env.example .env
uv run --no-sync --env-file .env uvicorn wlcv.api.app:app --host 127.0.0.1 --port 8000
```

기본 추론 장치는 CPU입니다. Mac GPU를 사용하려면 `.env`에 `WL_API_DEVICE=mps`를 지정하고 서버를 다시 시작하세요. 서버 한 프로세스가 영상 한 편씩 분석합니다. 터미널에서 Ctrl+C를 누르면 종료됩니다.

## 브라우저에서 사용

1. [API 문서 열기](http://127.0.0.1:8000/docs) → **POST /analyses** → **Try it out**을 누릅니다.
2. `video`에 MP4를 선택합니다. `near_side`는 카메라에 가까운 **선수의 왼쪽·오른쪽**, `facing`은 선수가 보는 **화면의 왼쪽·오른쪽**입니다. 각 값은 `left` 또는 `right`입니다.
3. 기본 `single`은 베이스라인으로 추적합니다. `cutmix`를 선택하면 기본 모델이 비운 프레임에 보조 모델을 사용합니다. 수동 시점은 아는 것만 입력하세요. 첫 프레임은 0입니다.
4. **Execute**를 누릅니다. 완료 응답의 `summary_url`과 `files`는 같은 서버의 경로입니다. 예를 들어 `http://127.0.0.1:8000` 뒤에 `files["overlay.mp4"]` 값을 붙여 브라우저 주소창에서 열 수 있습니다. `frames.csv`는 다운로드하고, 캐치·피니시 관측이 있으면 결과 PNG도 열 수 있습니다.

| 요청 | 반환 |
| --- | --- |
| `POST /analyses` | 완료 시 201, 분석 ID·요약·파일 링크. `Location` 헤더로도 요약 주소를 확인합니다. |
| `GET /analyses/{analysis_id}` | 저장된 요약과 파일 링크 |
| `GET /analyses/{analysis_id}/files/{filename}` | 오버레이 MP4, 프레임 CSV 또는 생성된 캐치→피니시 PNG |

`plate_counts`는 직접 검출·시간 연결·좌표 누락을 셉니다. 새 영상은 수동 정답이 없어 정확도·오탐률을 채점할 수 없습니다. `lift_to_knee_rise_px`는 이륙→무릎 통과, `knee_to_contact_rise_px`는 무릎 통과→접촉 상승량입니다. 수동 시점이나 좌표가 빠져 계산할 수 없으면 값은 `null`입니다. 후자의 측정 출처와 보류 이유도 함께 반환합니다.

## c04 재현

다른 터미널에서 프로젝트 루트를 열고 실행합니다.

```bash
curl -sS http://127.0.0.1:8000/analyses \
  -F 'video=@data/raw/videos/c04_sn_f_01.mp4' \
  -F near_side=left -F facing=left -F plate_mode=single \
  -F lift_off=11 -F knee_pass=27 -F drive=43 -F catch=78 -F finish=107
```

원본 업로드의 임시 사본은 요청 종료 시 정리됩니다. 결과는 `WL_OUTPUT_DIR/api/<분석 ID>/`에 저장되므로 서버를 다시 시작해도 같은 ID로 조회할 수 있습니다. API는 사용자가 올린 원본과 모델 경로가 포함된 로컬 `summary.json`을 다운로드 목록에서 제외합니다.

## Mac CPU에서 확인한 결과

| HTTP로 제출한 영상 | 결과 |
| --- | --- |
| c04 기본 모델, 수동 시점 5개 | 116장 모두 직접 검출. 이륙 11→무릎 통과 27 상승량 +160.514px. 무릎 통과→접촉은 코사인으로 +215.43px |
| a03 CutMix, 무릎 통과·접촉 지정 | 전체 122장. 동작 중 수동 박스 40장에서 단독 모델 27장, API 보조 모드 35장 정탐. 4~6번과 10~14번 복구 |
| c04 기본 모델, 시점 생략 | 116프레임 오버레이와 CSV 생성. 구간 상승량은 `null` |

시점을 지정한 두 영상의 CSV와 디코딩한 오버레이 238프레임을 같은 입력의 CLI 결과와 대조해 일치를 확인했습니다. c04의 캐치→피니시 그래프도 같은 픽셀로 저장됐습니다. 이 Mac의 로컬 HTTP에서 업로드 시작부터 요약 재조회 및 생성된 파일 다운로드까지 c04는 12.26초, a03은 13.59초였습니다. 각각 1회 측정한 시연 기록입니다.
