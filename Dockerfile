FROM python:3.11-slim AS base

LABEL maintainer="f1-2026-engine" \
      description="F1 2026 Strategy Simulation Engine"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "main.py"]
