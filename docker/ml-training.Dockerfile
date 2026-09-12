# syntax=docker/dockerfile:1.7
#
# Training image.
#
# Separate from the backend deliberately: training needs XGBoost, MLflow and
# (optionally) Torch, none of which belong in a latency-sensitive API image.
# It also runs on a different schedule and can be resourced independently.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/ml:/app/backend \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ml/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY ml /app/ml
COPY data-generation /app/data-generation
COPY backend/app /app/backend/app
COPY scripts /app/scripts

RUN useradd --create-home --uid 10002 trainer \
    && mkdir -p /app/ml/artifacts /app/data \
    && chown -R trainer:trainer /app
USER trainer

# No CMD that starts a server: this image is invoked as a job.
#   docker compose run --rm ml-training python ml/pipelines/train.py --mlflow
CMD ["python", "ml/pipelines/train.py", "--help"]
