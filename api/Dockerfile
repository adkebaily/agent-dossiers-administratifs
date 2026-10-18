FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-fra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/app ./api/app
COPY config ./config
ENV CONFIG_DIR=/srv/config

EXPOSE 8000
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
