FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python - <<'PY'
from pathlib import Path
src = Path('/tmp/requirements.txt')
dst = Path('/tmp/requirements-utf8.txt')
text = src.read_text(encoding='utf-16')
dst.write_text(text, encoding='utf-8')
PY

RUN pip install --no-cache-dir -r /tmp/requirements-utf8.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
