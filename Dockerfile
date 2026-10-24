FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# The anomaly model is trained on simulated baseline driving at build time,
# so the image always carries a model that matches its scikit-learn version.
RUN fuelsev train --out /app/models/anomaly.joblib

RUN useradd --create-home --uid 10001 fuelsev \
    && mkdir -p /data && chown fuelsev /data
USER fuelsev

ENV FUELSEV_DB_PATH=/data/fuelsev.db
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status==200 else 1)"

ENTRYPOINT ["fuelsev"]
CMD ["api", "--host", "0.0.0.0", "--port", "8000"]
