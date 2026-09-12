# syntax=docker/dockerfile:1.7
#
# Backend API image.
#
# Multi-stage: wheels are built in a stage that has compilers, and the runtime
# stage installs from those wheels. The runtime image therefore never contains
# gcc or the build headers - smaller, and a much smaller attack surface.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY backend/requirements.txt ./requirements.txt
RUN pip wheel --wheel-dir /wheels -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend:/app/ml

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

# Runs as a non-root user. A container that does not need root should not have
# it: a container escape from an unprivileged process is far less useful.
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /wheels /wheels
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-index --find-links=/wheels -r /tmp/requirements.txt \
    && rm -rf /wheels /tmp/requirements.txt

WORKDIR /app
COPY --chown=appuser:appuser backend /app/backend
COPY --chown=appuser:appuser ml/recsys /app/ml/recsys

USER appuser
EXPOSE 8000

# Hits readiness, not liveness: an instance with no model loaded is up but not
# useful, and should not receive traffic.
HEALTHCHECK --interval=20s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
