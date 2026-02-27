FROM node:20-bookworm-slim AS node

FROM python:3.12

WORKDIR /app

COPY --from=node /usr/local /usr/local

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync

COPY main.py .


CMD ["uv", "run", "./main.py"]