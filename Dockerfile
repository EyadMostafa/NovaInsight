# syntax=docker/dockerfile:1.7

# Stage 1: builder

FROM python:3.11-slim-bookworm AS builder

ARG EXTRAS=gemini

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build deps needed for wheels on non-amd64 arches (numpy/scipy/umap fallback paths)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only what pip needs to resolve deps first (better layer caching)
COPY pyproject.toml README.md ./
COPY sleuth ./sleuth

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip wheel \
 && pip install ".[$EXTRAS]"

# Stage 2: runtime

FROM python:3.11-slim-bookworm AS runtime

# matplotlib needs libgomp1 for OpenMP-linked numpy/scipy wheels on some arches
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r sleuth \
    && useradd -r -g sleuth -u 1000 -d /home/sleuth -m sleuth

COPY --from=builder /opt/venv /opt/venv
COPY --chown=sleuth:sleuth sleuth /app/sleuth
COPY --chown=sleuth:sleuth pyproject.toml /app/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Force headless matplotlib — belt-and-braces alongside the in-code call
    MPLBACKEND=Agg \
    SLEUTH_ENV=prod \
    # Redirect cache and output into the mounted /work volume
    SLEUTH_CACHE_DIRECTORY_PATH=/work/.sleuth_cache \
    SLEUTH_OUTPUT_DEFAULT_DIRECTORY=/work

WORKDIR /work
USER sleuth

LABEL org.opencontainers.image.title="sleuth" \
      org.opencontainers.image.description="Autonomous CLI data analysis tool" \
      org.opencontainers.image.source="https://github.com/EyadMostafa/sleuth" \
      org.opencontainers.image.licenses="MIT"

ENTRYPOINT ["sleuth"]
CMD ["--help"]
