ARG IMAGE_PLATFORM=linux/amd64
FROM --platform=${IMAGE_PLATFORM} python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
# The model is not in Git: run "python tools/fetch_model.py" once before building.
COPY models/ models/

# WHY non-root: the service only opens outgoing connections (camera, broker).
RUN useradd --system gesture
USER gesture

CMD ["python", "src/main.py"]
