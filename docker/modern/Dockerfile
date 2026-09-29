# MODERN_A image (PORTABILIDADE.md §3.1): the phase-A environment for the port of
# Graphix-T5 away from the legacy eyuansu62/graphix-text-to-sql:v2 image.
#   docker build -t graphix-modern:a1 -f docker/modern/Dockerfile docker/modern
# Everything is pinned: the base by digest, every added package by requirements-a1.lock
# (installed with --no-deps, so nothing outside the lock can slip in). Code and data are
# mounted at run time, never baked in, so one image serves every commit of phase A.

# pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime: Python 3.11.9, torch 2.4.0,
# CUDA 12.1, cuDNN 9.1.0, numpy 1.26.4, Ubuntu 22.04.
FROM pytorch/pytorch@sha256:68c022c2f4627943a6f3e574cfd2c8ae4256210d5f66ae2b117942e0a8d4fa9d

COPY requirements-a1.lock /opt/graphix/requirements-a1.lock
RUN pip install --no-cache-dir --no-deps -r /opt/graphix/requirements-a1.lock \
        -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html \
    && pip check

# DGLBACKEND avoids DGL writing a backend config under $HOME on first import.
ENV DGLBACKEND=pytorch \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.title="graphix-modern" \
      org.opencontainers.image.description="Graphix-T5 port, phase A (MODERN_A): torch 2.4.0+cu121, dgl 2.4.0, transformers 4.57.6" \
      graphix.stack="MODERN_A" \
      graphix.lock="requirements-a1.lock"

WORKDIR /app
