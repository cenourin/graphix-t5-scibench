# MODERN_A image (PORTABILIDADE.md §3.1): the phase-A environment for the port of
# Graphix-T5 away from the legacy eyuansu62/graphix-text-to-sql:v2 image.
#   docker build -t graphix-modern:a5 -f docker/modern/Dockerfile docker/modern
# (a1, built from commit 7c46c63 with requirements-a1.lock, is published as
#  silveirabruno/graphix-modern:a1 and kept for the A0-A5 evidence; a2 likewise.)
# Everything is pinned: bases by digest, every added package by requirements-a5.lock
# (installed with --no-deps, so nothing outside the lock can slip in), rapidfuzz by the
# sdist's sha256. Code and data are mounted at run time, never baked in.

# ---- build stage: rapidfuzz 2.0.5 (the legacy image's version) for Python 3.11 ----------
# No cp311 wheel exists, and its sdist ships Cython output that does not compile for 3.11
# (it uses CPython internals removed in 3.11). Regenerate it from the original .pyx with
# Cython 3.0.0b1 (the requested 3.0.0a10 is no longer on PyPI; Cython only generates the
# C API glue, the matching algorithms are rapidfuzz-cpp's C++) and build the wheel.
# pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel: same Python 3.11.9 as the runtime base.
FROM pytorch/pytorch@sha256:a55ff10111eb11f998884327d37361592e632899edd24fce99886b69289e33e6 AS rapidfuzz-build
RUN pip install --no-cache-dir "Cython==3.0.0b1" "scikit-build==0.13.1" "cmake==3.27.9" "ninja==1.11.1.1" \
        "rapidfuzz_capi==1.0.5" \
    && pip download --no-deps --no-binary :all: --no-build-isolation rapidfuzz==2.0.5 -d /src \
    && echo "216b49198574807aa7f68a02a357bfee9c8898c4336a462a16faf3b88f404054  /src/rapidfuzz-2.0.5.tar.gz" \
        | sha256sum -c - \
    && cd /src && tar xzf rapidfuzz-2.0.5.tar.gz && cd rapidfuzz-2.0.5 \
    && find src -name "*.cxx" -delete \
    && pip wheel --no-deps --no-build-isolation -w /wheels .

# ---- runtime image -----------------------------------------------------------------------
# pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime: Python 3.11.9, torch 2.4.0,
# CUDA 12.1, cuDNN 9.1.0, numpy 1.26.4, Ubuntu 22.04.
FROM pytorch/pytorch@sha256:68c022c2f4627943a6f3e574cfd2c8ae4256210d5f66ae2b117942e0a8d4fa9d

COPY requirements-a5.lock /opt/graphix/requirements-a5.lock
COPY --from=rapidfuzz-build /wheels /opt/graphix/wheels
RUN pip install --no-cache-dir --no-deps -r /opt/graphix/requirements-a5.lock \
        -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html \
    && pip install --no-cache-dir --no-deps /opt/graphix/wheels/rapidfuzz-2.0.5-*.whl \
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
      org.opencontainers.image.description="Graphix-T5 port, phase A (MODERN_A): torch 2.4.0+cu121, dgl 2.4.0, transformers 4.57.6, accelerate 1.3.0, rapidfuzz 2.0.5; evaluators, optuna" \
      graphix.stack="MODERN_A" \
      graphix.lock="requirements-a5.lock"

WORKDIR /app
