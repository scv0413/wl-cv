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
