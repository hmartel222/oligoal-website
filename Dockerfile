FROM python:3.11-slim-bookworm

ENV PYTHONPATH=/srv/oligoal/src:/srv/oligoal/vendor/OligoGym \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PORT=8080

WORKDIR /srv/oligoal

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 oligoal \
    && install -d -o oligoal -g oligoal vendor artifacts

COPY requirements-website.txt ./
RUN python -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.3.1+cpu" \
    && python -m pip install --no-cache-dir -r requirements-website.txt

COPY scripts/setup-oligogym.sh scripts/

USER oligoal
RUN sh scripts/setup-oligogym.sh

COPY app/ app/
COPY src/ src/
COPY data/oligogym/ data/oligogym/

EXPOSE 8080
CMD ["sh", "-c", "exec python app/server.py --host 0.0.0.0 --port \"$PORT\""]
