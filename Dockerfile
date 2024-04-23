FROM python:3.10-alpine as base

FROM base as builder-image
COPY requirements.lock ./
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -r requirements.lock

WORKDIR /app
COPY src .
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn","function_server.main:app"]