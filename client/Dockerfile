FROM --platform=$BUILDPLATFORM python:3.10-alpine AS builder

WORKDIR /client

RUN apk update && apk add --no-cache git

COPY client/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3", "app.py"]
