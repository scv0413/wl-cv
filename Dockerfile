FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /usr/local/bin/uv
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/cpu/pyproject.toml deploy/cpu/uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

COPY wlcv/__init__.py wlcv/config.py ./wlcv/
COPY wlcv/api/*.py ./wlcv/api/
COPY wlcv/runtime/*.py ./wlcv/runtime/
COPY wlcv/data/__init__.py wlcv/data/mp4_input.py wlcv/data/frames.py ./wlcv/data/
ENV PATH="/app/.venv/bin:$PATH" \
    WL_MODEL_DIR=/models WL_OUTPUT_DIR=/results WL_API_DEVICE=cpu WL_VIDEO_DECODER=pyav \
    MPLBACKEND=Agg YOLO_OFFLINE=true YOLO_AUTOINSTALL=false
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "wlcv.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
