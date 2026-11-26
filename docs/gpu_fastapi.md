# RTX 3090에서 합성 MP4로 FastAPI 검증

검증할 경로는 `POST /analyses`에서 시작해 TensorRT 추론, 원판 시간 연결, H.264 오버레이와 프레임별 CSV 다운로드로 이어진다. FFmpeg의 색상·도형 패턴으로 만든 세로 MP4를 사용한다. 합성 장면의 검출 수로 선수 영상의 정확도를 평가할 수는 없다.

## 모델 준비

`deploy/gpu/release.json`은 기존 검증에 사용한 ONNX 네 개와 원판 엔진 세 개의 SHA-256이다. 모델 원본은 별도로 공급한다. 입력 폴더에는 다음 파일만 필요하다.

| 역할 | ONNX 파일 | 재사용할 수 있는 RTX 3090 엔진 |
| --- | --- | --- |
| 원판 주 모델 | `baseline.onnx` | `baseline-fp16.engine` |
| CutMix 보조 | `cutmix.onnx` | `cutmix-fp16.engine` |
| 무릎 통과→접촉 상승량 | `cosine.onnx` | `cosine-fp16.engine` |
| 가까운 쪽 자세 | `pose.onnx` | 대상 장치에서 빌드 |

준비 스크립트가 파일 체크섬을 대조한다. 캐시된 엔진을 역직렬화할 수 없으면 해당 ONNX에서 다시 빌드한다. 입력 형상은 배치 1, RGB 640×384(높이×너비)이고, 입출력 자료형은 FP32다. FP16 사용 여부는 엔진 검사기로 내부 층의 출력 자료형을 확인한다. 준비된 엔진과 검사 결과 `engines.json`이 같은 폴더에 저장된다.

## GPU 컨테이너 실행

Linux AMD64, 호환 NVIDIA 드라이버와 GPU를 컨테이너에 연결한 환경에서 프로젝트 루트를 연다. 아래 명령은 모델 소스가 `./models/gpu_source/`에 있는 경우다. `outputs/gpu_release/models/tensorrt`는 이번 준비 명령으로 새로 생성한다.

```bash
docker build -f deploy/gpu/Dockerfile -t wlcv-api:trt .
mkdir -p outputs/gpu_release/models outputs/gpu_release/results
docker run --rm --gpus device=0 \
  -v "$PWD/models/gpu_source:/source:ro" \
  -v "$PWD/outputs/gpu_release/models:/models" \
  wlcv-api:trt python deploy/gpu/prepare_engines.py /source /models/tensorrt

docker run -d --name wlcv-trt-api --gpus device=0 \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/outputs/gpu_release/models:/models:ro" \
  -v "$PWD/outputs/gpu_release/results:/results" wlcv-api:trt
docker exec wlcv-trt-api python deploy/gpu/smoke.py /results/synthetic-01
```

이미 PyTorch 2.8.0 + CUDA 12.8 컨테이너로 실행 중인 RunPod에서는 같은 소스와 `deploy/gpu/requirements.txt`를 사용할 수 있다. FFmpeg와 ffprobe가 필요하다. 아래 명령은 Pod의 프로젝트 폴더에서 실행한다. API는 Pod 내부의 루프백 주소에만 연다.

```bash
python -m pip install -r deploy/gpu/requirements.txt
export WL_MODEL_DIR="$PWD/release/models" WL_OUTPUT_DIR="$PWD/release/results"
export WL_API_BACKEND=tensorrt WL_API_DEVICE=0 NVIDIA_TF32_OVERRIDE=0
export WL_VIDEO_DECODER=pyav PYAV_LOGGING=off MPLBACKEND=Agg YOLO_OFFLINE=true YOLO_AUTOINSTALL=false
python deploy/gpu/prepare_engines.py model_source "$WL_MODEL_DIR/tensorrt"
python -m uvicorn wlcv.api.app:app --host 127.0.0.1 --port 8000
```

다른 터미널에서 `python deploy/gpu/smoke.py release/results/synthetic-01`을 실행한다. 출력 폴더에는 실제 HTTP 응답, 합성 입력, 내려받은 오버레이·CSV와 검증 JSON이 남는다. 검증 프로그램은 201 응답과 저장된 요약을 대조하고 네 모델 모두 실제로 호출한 백엔드가 `TensorRTBackend`인지 확인한다. 4·12번을 시험용 동작 시점으로 입력해 코사인 경로까지 실행한다. 오버레이가 H.264로 끝까지 디코딩되고 자막 영역이 달라졌는지, CSV에 원본 프레임 번호 0~23이 순서대로 저장됐는지도 확인한다.

## Mac에서 확인한 범위

동일한 합성 MP4를 Mac CPU의 FastAPI에 제출했다. 201 응답을 받았고 H.264 오버레이 24프레임과 프레임 번호 0~23에 대응하는 CSV 24행을 다운로드해 읽었다. 기존 API 테스트 4개도 통과했다. 네 ONNX의 입출력 형상과 체크섬, 원판 엔진 세 개의 원본 바이트 보존을 확인했다. GPU에서는 별도 실행의 `engines.json`과 `verification.json`이 내부 FP16 사용과 실제 백엔드 호출의 증거가 된다.

## RTX 3090에서 확인한 결과

2026-11-26 RunPod Community RTX 3090 24GB에서 합성 MP4 한 편(384×640, 24fps, 24프레임)을 제출했다. RunPod의 PyTorch 2.8.0+cu128 컨테이너 안에서 저장소에 고정한 패키지와 TensorRT 10.13.3.9를 사용했다. 이 실행은 해당 컨테이너에서 FastAPI를 구동한 검증이며, 저장소 GPU Dockerfile의 별도 이미지 빌드는 검증 범위에 포함되지 않았다.

| 검증 항목 | 실측 결과 |
| --- | --- |
| 업로드부터 분석 응답까지 | 첫 요청 2.304초, HTTP 201. 모델 로드·영상 처리·결과 저장 포함 |
| 저장 결과 다운로드 | 요약·오버레이·프레임 CSV 모두 HTTP 200 |
| 오버레이 | H.264, 384×640, 24프레임 디코딩. 전체 24프레임의 자막 영역 변화 확인 |
| 프레임 CSV | 24행, 원본 프레임 번호 0~23 순서 일치 |
| 실제 추론 | 베이스라인·CutMix·코사인·자세 모두 CUDA의 `TensorRTBackend` |

기존 원판 엔진 3개를 다시 로드하고 자세 엔진은 해당 GPU에서 빌드했다. 엔진 검사에서 FP16 출력을 쓰는 층은 베이스라인 234개, CutMix 220개, 코사인 210개, 자세 229개였다. Linux OpenCV의 H.264 인코더 초기화가 실패한 뒤 기존 FFmpeg/libx264 대체 경로로 영상이 저장됐다.

합성 패턴의 24프레임 모두에서 최종 원판 좌표는 베이스라인에서 선택됐다. 이 결과로 확인한 범위는 요청 처리, 네 엔진 호출, 결과물 저장과 다운로드다. 선수 원판의 실제 검출 정확도와 CutMix의 복구율은 선수 영상 검증 결과에서 판단한다.

## 저장소 Dockerfile로 빌드한 이미지 검증

비공개 GitHub Actions에서 `deploy/gpu/Dockerfile`을 Linux AMD64로 빌드했다. Community RTX 3090 Pod에 다음 다이제스트를 지정했고, 이미지의 기본 명령으로 FastAPI가 시작됐다. 네 TensorRT 엔진은 실행할 때 별도로 전달했다. [이미지 빌드 기록](https://github.com/scv0413/wl-cv/actions/runs/45820280308)

```text
ghcr.io/scv0413/wl-cv-gpu-api@sha256:d815f96d00f5f20ec15cf03ed5dab1de3c84e107ff7ad1e980ba7c8d105e3e2a
```

직전 RunPod PyTorch 컨테이너에서 사용한 합성 MP4와 네 엔진의 SHA-256을 대조했다. 두 번 모두 같은 영상에 무릎 통과 4번과 접촉 12번을 지정했다.

| 확인 항목 | RunPod PyTorch 컨테이너 | 저장소 Dockerfile 이미지 |
| --- | --- | --- |
| 첫 분석 요청 | HTTP 201, 2.304초 | HTTP 201, 2.011초 |
| 실제 추론 백엔드 | TensorRT 엔진 4개 | TensorRT 엔진 4개 |
| 다운로드한 오버레이 | H.264, 384×640, 24프레임 | 같은 규격, Mac에서 24프레임 전체 디코딩 |
| 다운로드한 프레임 CSV | 24행, 20열 | 프레임 번호 0~23, 기준 CSV와 바이트 단위 일치 |

Pod 생성부터 Uvicorn 시작까지 약 7분 45초가 걸렸다. 여기에는 비공개 이미지 다운로드와 압축 해제가 포함된다. 표의 첫 요청 시간은 API가 기동한 뒤 영상을 보내 201 응답을 받기까지의 시간이다. 환경별로 한 번씩 측정해 처리 속도의 우열은 판단하지 않았다. 합성 패턴에서 실제 선수에 대한 정확도는 평가하지 않았다.

같은 입력으로 재현하려면 API를 띄운 뒤 검증 클라이언트에 이전 합성 MP4 경로와 새 결과 폴더를 전달한다. 클라이언트는 전송한 파일의 SHA-256을 `verification.json`에 기록한다.

```bash
python deploy/gpu/smoke.py ./outputs/gpu_replay --input ./outputs/previous/synthetic.mp4
```
