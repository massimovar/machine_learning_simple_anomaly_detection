# ==========================================================================
# Dockerfile — Builds the anomaly detection container
# ==========================================================================
#
# TARGET:
#   ARM64 (edge device): podman compose up --build
#
# This Dockerfile is intentionally arm64-only for the edge deployment target.
# It installs tflite-runtime for lightweight inference and NXP delegate support.
#
# HOW TO EXPORT the image for the edge device (Portainer):
#   podman save -o anomaly_detection-detector.tar \
#               anomaly_detection-detector:latest
#
# NPU ACCELERATION (NXP i.MX 8M Plus):
#   Currently disabled.  To enable, set use_npu: true in config.yaml
#   once the NXP eIQ runtime (libvx_delegate.so) is present on the device.
#   No rebuild required.
# ==========================================================================

# --platform is injected by podman compose from the `platform:` key in
# docker-compose.yml. We keep the fallback as arm64 to match the only target.
FROM --platform=${TARGETPLATFORM:-linux/arm64} python:3.11-slim

# --- Working directory -------------------------------------------------------
WORKDIR /app

# --- Install Python dependencies ---------------------------------------------
COPY requirements.txt .

# Install the TFLite inference backend for arm64.
RUN pip install --no-cache-dir "tflite-runtime>=2.14,<2.15"

# Install the remaining runtime dependencies (paho-mqtt, numpy, sklearn, pyyaml).
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy project files ------------------------------------------------------
COPY config.yaml .
COPY src/ src/
COPY models/ models/

# --- Default command ---------------------------------------------------------
CMD ["python", "src/main.py"]
