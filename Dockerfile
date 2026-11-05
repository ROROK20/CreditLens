# CreditLens backend — FastAPI + logistic scorer + SHAP, served by uvicorn on 8002.
# The app trains a small synthetic model on startup, so the container is ready
# to score with zero manual steps.
# Python 3.12 is a hard floor, not a preference: the pinned dependency set in
# requirements.txt (shap==0.52.0 in particular) declares python_requires >= 3.12,
# so a 3.11 base image fails this build at dependency resolution.
FROM python:3.12-slim

# Build deps for xgboost/shap (libgomp) — slim image lacks them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the Python package and its runtime deps first (better layer caching).
COPY requirements.txt setup.py README.md ./
COPY creditlens ./creditlens
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

EXPOSE 8002

# host 0.0.0.0 so the service is reachable from outside the container. Honor an
# injected $PORT (Render/Cloud Run set this) but default to 8002 locally.
ENV PORT=8002
CMD ["sh", "-c", "uvicorn creditlens.api.app:app --host 0.0.0.0 --port ${PORT:-8002}"]
