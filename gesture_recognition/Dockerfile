# ============================================================================
# Dockerfile — gesture recognition service (laptop amd64 and OptixEdge arm64)
# ============================================================================
# One image for every target:
#   laptop          podman compose -f docker-compose.yml up --build
#   OptixEdge       podman build --platform linux/arm64 -t localhost/gesture_recognition-gesture:edge .
#                   (deploy_tools/deploy_edge.ps1 does it and ships the image)
#
# CPU or NPU is decided at run time, not at build time:
#   - CPU: the image's own ai-edge-litert (XNNPACK).  Always available.
#   - NPU: docker-compose.npu.yml mounts the device's /usr/lib read-only and
#     passes /dev/galcore; tools/npu_entrypoint.sh borrows only the NPU
#     libraries and the device's tflite_runtime (same pairing as Part 1).
#     If anything is missing, src/pose.py falls back to the CPU.
#
# WHY Debian trixie + Python 3.12 (same rules as anomaly_detection/Dockerfile.npu):
#   - the borrowed NPU libraries were built against the device's glibc 2.39;
#     glibc is backward compatible only, so the image needs >= 2.39 (trixie: 2.41);
#   - the borrowed tflite_runtime is a compiled CPython 3.12 module.
# ============================================================================
FROM python:3.12-slim-trixie

WORKDIR /app

COPY requirements.txt .
# WHY patchelf: glibc >= 2.41 refuses to load a library marked as needing an
# executable stack ("cannot enable executable stack"); the LiteRT wheel has
# that mark by accident.  Clearing it is safe and makes the image safer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends patchelf \
 && pip install --no-cache-dir -r requirements.txt \
 && find /usr/local/lib/python3*/site-packages/ai_edge_litert -name "*.so*" -type f \
        -exec patchelf --clear-execstack {} + \
 && apt-get purge -y patchelf && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

COPY config.yaml .
COPY src/ src/
# Only the model: tools/fetch_model.py downloads it once before building.
COPY models/movenet_lightning_int8.tflite models/
COPY tools/npu_entrypoint.sh tools/
# WHY sed: a Windows checkout may add CR line endings, which /bin/sh rejects.
RUN sed -i 's/\r$//' tools/npu_entrypoint.sh

# WHY non-root: the service needs no privileges.  It owns the folders the
# entrypoint (borrowed NPU libraries) and OpenVX (compiled graph cache) write to.
RUN useradd --system gesture \
 && mkdir -p /opt/npu /var/cache/npu \
 && chown gesture /opt/npu /var/cache/npu
USER gesture

# WHY TCP: RTP over UDP does not cross the container NAT (rootless Docker on
# the OptixEdge, the Podman VM on the laptop); RTSP interleaved over TCP does.
ENV PYTHONUNBUFFERED=1 \
    OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"

EXPOSE 8081
ENTRYPOINT ["sh", "tools/npu_entrypoint.sh"]
CMD ["python", "src/main.py"]