FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-nanum \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/gpu/requirements.txt deploy/gpu/requirements.txt
RUN python -m pip install --no-cache-dir -r deploy/gpu/requirements.txt
COPY deploy/gpu/*.py deploy/gpu/release.json deploy/gpu/
COPY wlcv/__init__.py wlcv/config.py ./wlcv/
COPY wlcv/api/*.py ./wlcv/api/
COPY wlcv/runtime/*.py ./wlcv/runtime/
COPY wlcv/data/__init__.py wlcv/data/frames.py wlcv/data/mp4_input.py ./wlcv/data/
ENV WL_MODEL_DIR=/models WL_OUTPUT_DIR=/results WL_API_BACKEND=tensorrt WL_API_DEVICE=0 \
    NVIDIA_TF32_OVERRIDE=0 WL_VIDEO_DECODER=pyav PYAV_LOGGING=off MPLBACKEND=Agg \
    YOLO_OFFLINE=true YOLO_AUTOINSTALL=false
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "wlcv.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
