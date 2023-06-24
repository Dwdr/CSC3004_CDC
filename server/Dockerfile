# syntax=docker/dockerfile:1.4
FROM python:3.10 AS builder

WORKDIR /server

RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-dev ffmpeg

COPY server/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY server .

EXPOSE 8001
EXPOSE 30081

CMD ["python3", "app.py"]
