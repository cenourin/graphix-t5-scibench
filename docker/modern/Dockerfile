# MODERN_A image (PORTABILIDADE.md §3.1): the phase-A environment for the port of
# Graphix-T5 away from the legacy eyuansu62/graphix-text-to-sql:v2 image.
#   docker build -t graphix-modern:a4 -f docker/modern/Dockerfile docker/modern
# (a1, built from commit 7c46c63 with requirements-a1.lock, is published as
#  silveirabruno/graphix-modern:a1 and kept for the A0-A5 evidence.)
# Everything is pinned: the base by digest, every added package by requirements-a4.lock
# (installed with --no-deps, so nothing outside the lock can slip in). Code and data are
# mounted at run time, never baked in, so one image serves every commit of phase A.

# pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime: Python 3.11.9, torch 2.4.0,
# CUDA 12.1, cuDNN 9.1.0, numpy 1.26.4, Ubuntu 22.04.
FROM pytorch/pytorch@sha256:68c022c2f4627943a6f3e574cfd2c8ae4256210d5f66ae2b117942e0a8d4fa9d

COPY requirements-a4.lock /opt/graphix/requirements-a4.lock
RUN pip install --no-cache-dir --no-deps -r /opt/graphix/requirements-a4.lock \
        -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html \
    && pip check

# nltk punkt data for the evaluators' word_tokenize, byte-identical to the legacy image's
# /usr/local/share/nltk_data/tokenizers/punkt.zip (sha256 9a74e3cc...), at the same path.
COPY nltk_data /usr/local/share/nltk_data
# nltk 3.7 loads punkt's pickles from the extracted directory, as in the legacy image,
# which ships both the zip and its extraction.
RUN python -m zipfile -e /usr/local/share/nltk_data/tokenizers/punkt.zip /usr/local/share/nltk_data/tokenizers/

# DGLBACKEND avoids DGL writing a backend config under $HOME on first import.
ENV DGLBACKEND=pytorch \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.title="graphix-modern" \
      org.opencontainers.image.description="Graphix-T5 port, phase A (MODERN_A): torch 2.4.0+cu121, dgl 2.4.0, transformers 4.57.6; evaluators, optuna, rapidfuzz, accelerate" \
      graphix.stack="MODERN_A" \
      graphix.lock="requirements-a4.lock"

WORKDIR /app
