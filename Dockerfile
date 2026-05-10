FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# OpenCV runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cache layer)
COPY .python-version pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY app.py ./

EXPOSE 7860

CMD ["uv", "run", "python", "app.py"]
