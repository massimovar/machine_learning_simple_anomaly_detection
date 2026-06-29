FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir "tflite-runtime>=2.14,<2.15"
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
COPY models/ models/

CMD ["python", "src/main.py"]
