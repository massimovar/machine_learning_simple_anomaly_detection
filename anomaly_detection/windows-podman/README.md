# Windows Podman Compose

This folder contains a dedicated local container setup for running the anomaly detection stack on Windows with Podman.

The root compose file in `anomaly_detection/docker-compose.yml` is an edge deployment definition. It targets `linux/arm64` and assumes an ARM64 detector image. That is correct for the edge device, but it is not correct for a Windows Podman machine running a Linux `amd64` VM.

## Why this folder exists

On Windows, the original compose hit two separate problems:

1. The detector service was pinned to `linux/arm64`, so Podman on an `amd64` machine failed with `Exec format error`.
2. Reusing generic local image names made it easy for Podman to pick up previously cached images with the wrong architecture.

This folder solves that by providing a separate local-only stack that builds and runs `linux/amd64` images explicitly.

## What is in this folder

- `docker-compose.yml`
  - Windows-specific compose definition
  - forces `platform: linux/amd64`
  - uses distinct local image tags so it does not collide with the ARM64 edge images
- `Dockerfile.detector`
  - builds the detector image for local Windows Podman use
  - installs `tflite-runtime` on `amd64`
  - falls back to TensorFlow only if `tflite-runtime` cannot be installed
- `Dockerfile.mqtt-broker`
  - builds a local Mosquitto broker image for `amd64`
- `mosquitto.conf`
  - exposes MQTT on `1883`
  - exposes WebSockets on `9001`

## Important supporting change outside this folder

The root `.dockerignore` in `anomaly_detection/.dockerignore` excludes the local Python virtual environment.

That change matters because Podman was previously trying to send the Windows `venv/` directory into the Linux build context, which caused build failures such as:

`archive/tar: missed writing ...`

Without that ignore rule, the detector image build is not reliable.

## Image names used here

This setup intentionally uses different local tags from the edge compose:

- `localhost/anomaly_mqtt_broker:windows-amd64`
- `localhost/anomaly_detector:windows-amd64`

That avoids reusing cached ARM64 images by mistake.

## How it works

The detector container still talks to the broker on the internal Compose network using:

- host name: `mqtt-broker`
- port: `1883`

The host port mapping only affects how your Windows host reaches the broker. It does not change how the detector reaches the broker inside the Compose network.

## Start the stack

Run these commands from the `anomaly_detection` folder.

### Option 1: Default host ports

Use this if nothing else is already using ports `1883` and `9001`.

```powershell
podman compose -f windows-podman/docker-compose.yml up -d --build
```

### Option 2: Alternate host ports

Use this if another MQTT broker or an older stack is already bound to `1883` or `9001`.

```powershell
$env:MQTT_HOST_PORT = "1884"
$env:MQTT_WS_HOST_PORT = "9002"
podman compose -f windows-podman/docker-compose.yml up -d --build
```

In that example:

- host MQTT port becomes `1884`
- host WebSocket port becomes `9002`
- container-internal ports remain `1883` and `9001`

## Stop the stack

```powershell
podman compose -f windows-podman/docker-compose.yml down
```

If you started it with alternate host ports in the same shell session, keep the same environment variables set when you run `down`.

## Verify that it started correctly

Check container state:

```powershell
podman ps -a --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}"
```

Check detector logs:

```powershell
podman logs anomaly_detection_windows-detector-1 --tail 50
```

Expected detector log sequence includes:

- loading `config.yaml`
- loading model, scaler, and threshold
- connecting to `mqtt-broker:1883`
- subscribing to `ftoptix/paperclip/sensors`

## Troubleshooting

### `Exec format error`

Cause:

- the ARM64 edge compose was started on an `amd64` Podman machine
- or Podman reused an old image with the wrong architecture

Fix:

- use this folder's compose file
- rebuild explicitly:

```powershell
podman compose -f windows-podman/docker-compose.yml build --no-cache
```

### `bind: address already in use`

Cause:

- another local process or another container stack is already using `1883` or `9001`

Fix:

- stop the conflicting stack
- or use alternate host ports:

```powershell
$env:MQTT_HOST_PORT = "1884"
$env:MQTT_WS_HOST_PORT = "9002"
podman compose -f windows-podman/docker-compose.yml up -d
```

### Detector build fails while tarring the build context

Cause:

- a local virtual environment was included in the Docker build context

Fix:

- make sure `venv/` is excluded in `anomaly_detection/.dockerignore`
- rebuild:

```powershell
podman compose -f windows-podman/docker-compose.yml build detector
```

## Relationship to the edge compose

Keep both definitions for different targets:

- `anomaly_detection/docker-compose.yml`
  - edge deployment
  - ARM64
- `anomaly_detection/windows-podman/docker-compose.yml`
  - local Windows Podman development and testing
  - AMD64

Do not replace the edge compose with this one. They solve different problems.