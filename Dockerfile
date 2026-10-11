
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface

WORKDIR /app

# Install the runtime library required by numerical dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /models/huggingface \
    && chown -R appuser:appuser /app /models /home/appuser

# Install Python dependencies.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application code and the existing vector store.
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser data/processed/faiss_store/ ./data/processed/faiss_store/

# Run the application as a non-root user.
USER appuser

EXPOSE 8000 8501
