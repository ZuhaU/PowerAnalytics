FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Playwright's browser binary is only needed for live SBP ingestion
RUN playwright install --with-deps chromium

COPY src /app/src
COPY Data /app/Data

RUN mkdir -p /app/logs

# USE_LIVE_INGESTION, ENABLE_AZURE_UPLOADS and ENABLE_AI_SUMMARY all
# default to "false" inside src/pipeline.py / generate_insights.py, so
# the container runs fully locally unless you pass them in at
# `docker run` time (see README for details). Note: ENABLE_AI_SUMMARY
# requires an Ollama installation reachable from inside the container,
# which this image does not include -- run the AI summary step on the
# host instead if you need it alongside a Dockerized pipeline.
CMD ["python", "src/pipeline.py"]
