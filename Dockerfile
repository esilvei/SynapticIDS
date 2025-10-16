# ---- Build Stage ----
FROM python:3.12 as builder

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv venv /app/.venv && \
    uv sync --python /app/.venv/bin/python

# ---- Final Stage ----
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 git netcat-openbsd && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv ./.venv

COPY ./requirements.txt /app/requirements.txt

COPY ./mlruns /app/mlruns

COPY ./src /app/src

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "src.synaptic_ids.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
