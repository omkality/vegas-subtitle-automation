# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df AS tools
ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp
WORKDIR /app
COPY requirements/audio.lock ./requirements/audio.lock
RUN pip install --no-cache-dir --only-binary=:all: --require-hashes -r requirements/audio.lock
COPY src ./src
COPY tests ./tests
COPY configs ./configs
RUN mkdir -p local && chown 1000:1000 local
USER 1000:1000
ENTRYPOINT ["python", "-m", "vegas_subtitles"]
CMD ["--help"]

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc AS gpu
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/local/cache/huggingface \
    PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY requirements/asr.lock ./requirements/asr.lock
RUN pip install --no-cache-dir --only-binary=:all: --require-hashes -r requirements/asr.lock
COPY src ./src
COPY tests ./tests
COPY configs ./configs
RUN mkdir -p local && chown 1000:1000 local
USER 1000:1000
ENTRYPOINT ["python", "-m", "vegas_subtitles"]
CMD ["--help"]
