FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching
COPY .python-version pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# Copy application source
COPY src/ ./src/
COPY configs/ ./configs/
COPY app.py ./

EXPOSE 7860

CMD ["uv", "run", "python", "app.py"]
