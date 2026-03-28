FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY pyproject.toml .
COPY backtest_api/ backtest_api/
COPY paper/ paper/
COPY scripts/ scripts/
RUN uv pip install --system .

EXPOSE 8070

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8070/health || exit 1

CMD ["uvicorn", "backtest_api.main:app", "--host", "0.0.0.0", "--port", "8070"]
