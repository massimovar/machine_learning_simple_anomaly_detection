<#
.SYNOPSIS
    Build the arm64 gesture image on this Windows PC and run it on the OptixEdge
    through the device's Docker API (tcp://<ip>:2375, LAN side).

.DESCRIPTION
    Works from any folder. The device has no internet: the image is built here
    (QEMU emulation) and uploaded as a file.

      Deploy  build + save + upload the image, start the service, show the log (default)
      Logs    follow the service log (Ctrl+C stops following)
      Status  container state, last log lines, MJPEG page reachable?
      Down    stop and remove the container (image and NPU cache volume are kept)
      Remove  Down + delete the image and the NPU cache volume from the device

    -SkipBuild  do not build/upload (image already on the device): only 'up'.
    -Npu        add docker-compose.npu.yml (/dev/galcore + /usr/lib). Only on a
                device with the NPU enabled: without /dev/galcore Docker refuses
                to create the container.
    -Engine     'podman' (default) or 'docker' (Docker Desktop) for the local build.
    -MqttHost   broker address as seen from the device (default: -DeviceIp).

    Needs gesture_recognition\.env with the camera address (never committed):
      CAMERA_URL=rtsp://user:password@192.168.8.66:554/stream2

    WARNING: port 2375 is the Docker API without TLS or password: whoever reaches
    it controls the device. Keep it on the LAN of a test bench only.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy_tools\deploy_edge.ps1
.EXAMPLE
    .\deploy_tools\deploy_edge.ps1 -SkipBuild -Npu      # switch an NPU device to the NPU
.EXAMPLE
    .\deploy_tools\deploy_edge.ps1 -Action Logs -DeviceIp 10.0.0.42
#>
param(
    [ValidateSet('Deploy', 'Logs', 'Status', 'Down', 'Remove')]
    [string]$Action = 'Deploy',
    [string]$DeviceIp = '10.0.0.42',
    [string]$MqttHost = '',
    [ValidateSet('podman', 'docker')]
    [string]$Engine = 'podman',
    [switch]$SkipBuild,
    [switch]$Npu
)

$ErrorActionPreference = 'Stop'

$Image = 'localhost/gesture_recognition-gesture:edge'
$Tar   = 'gesture_recognition-edge.tar'
$Api   = "http://${DeviceIp}:2375"
if (-not $MqttHost) { $MqttHost = $DeviceIp }

function Invoke-Step([string]$StepTitle, [scriptblock]$StepBlock) {
    Write-Host "`n==> $StepTitle" -ForegroundColor Cyan
    & $StepBlock
    if ($LASTEXITCODE -ne 0) { throw "Failed (exit code $LASTEXITCODE): $StepTitle" }
}

# True if the command exits with 0. Output is hidden; stderr must not become an error.
function Test-Quiet([scriptblock]$Probe) {
    $ErrorActionPreference = 'Continue'
    & $Probe *> $null
    return ($LASTEXITCODE -eq 0)
}

# docker-compose against the device. DOCKER_HOST and MQTT_HOST only for this call;
# CAMERA_URL comes from .env (docker-compose reads it from this folder).
function Invoke-Compose([string[]]$ComposeArgs, [switch]$WithNpu) {
    $files = @('-f', 'docker-compose.edge.yml')
    if ($Npu -or $WithNpu) { $files += '-f', 'docker-compose.npu.yml' }
    $saved = $env:DOCKER_HOST, $env:MQTT_HOST
    $env:DOCKER_HOST = "tcp://${DeviceIp}:2375"
    $env:MQTT_HOST = $MqttHost
    try {
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            & docker compose -p gesture_recognition @files @ComposeArgs
        } else {
            & docker-compose -p gesture_recognition @files @ComposeArgs
        }
    } finally {
        $env:DOCKER_HOST, $env:MQTT_HOST = $saved
    }
}

function Build-Image {
    if (-not (Get-Command $Engine -ErrorAction SilentlyContinue)) { throw "$Engine not found." }
    if (-not (Test-Path 'models\movenet_lightning_int8.tflite')) {
        throw 'Missing models\movenet_lightning_int8.tflite: run  python tools\fetch_model.py  first.'
    }
    if ($Engine -eq 'podman') {
        if (-not (Test-Quiet { podman info })) {
            Invoke-Step 'Starting the Podman machine' { podman machine start }
        }
        # The arm64 build runs under QEMU; the binfmt setting is lost at every machine restart.
        if (-not (Test-Quiet { podman run --rm --platform linux/arm64 docker.io/library/alpine:3.20 uname -m })) {
            Invoke-Step 'Enabling arm64 emulation (binfmt)' {
                podman run --rm --privileged docker.io/tonistiigi/binfmt --install arm64
            }
        }
        Invoke-Step 'Building the arm64 image (10-20 min the first time)' {
            podman build --platform linux/arm64 -t $Image .
        }
    } else {
        if ($env:DOCKER_HOST) {
            throw "DOCKER_HOST is set to '$env:DOCKER_HOST': the build must run on this PC. Run 'Remove-Item Env:DOCKER_HOST' first."
        }
        if (-not (Test-Quiet { docker info })) { throw 'Docker is not running: start Docker Desktop.' }
        Invoke-Step 'Building the arm64 image (10-20 min the first time)' {
            docker buildx build --platform linux/arm64 -t $Image --load .
        }
    }
    Remove-Item $Tar -ErrorAction SilentlyContinue
    Invoke-Step "Saving the image to $Tar" { & $Engine save -o $Tar $Image }
}

function Send-Image {
    $mb = [math]::Round((Get-Item $Tar).Length / 1MB)
    # -T streams the file instead of loading it in memory; 'Expect:' skips a 1 s wait.
    Invoke-Step "Uploading $Tar ($mb MB) to $Api" {
        curl.exe -sS -f -X POST -T $Tar -H 'Content-Type: application/x-tar' -H 'Expect:' "$Api/images/load"
    }
    Remove-Item $Tar
}

function Show-Stream {
    $url = "http://${DeviceIp}:8081/"
    try {
        $null = Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 5
        Write-Host "`nMJPEG page OK: $url  (URL for the FT Optix WebBrowser object)" -ForegroundColor Green
    } catch {
        Write-Host "`nMJPEG page not reachable yet: $url  (try -Action Status in a minute)" -ForegroundColor Yellow
    }
}

Push-Location (Split-Path $PSScriptRoot -Parent)   # the gesture_recognition folder
try {
    if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) { throw 'curl.exe not found.' }
    if (-not (Get-Command docker -ErrorAction SilentlyContinue) -and
        -not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        throw 'Neither docker nor docker-compose found (Podman Desktop installs docker-compose).'
    }
    if (-not (Test-Quiet { curl.exe -sS -f -m 5 "$Api/_ping" })) {
        throw "Docker API not reachable at $Api (LAN cable? API enabled on the device?)."
    }
    if (-not (Select-String -Path '.env' -Pattern '^\s*CAMERA_URL=' -Quiet -ErrorAction SilentlyContinue)) {
        throw 'Put CAMERA_URL=rtsp://user:password@<camera-ip>:554/stream2 in gesture_recognition\.env'
    }

    switch ($Action) {
        'Deploy' {
            if (-not $SkipBuild) { Build-Image; Send-Image }
            Invoke-Step "Starting the service (NPU: $([bool]$Npu), MQTT broker: $MqttHost)" {
                Invoke-Compose @('up', '-d', '--no-build', '--force-recreate')
            }
            Write-Host 'Waiting 20 s for the camera and the model...'
            Start-Sleep -Seconds 20
            Invoke-Step 'Service log' { Invoke-Compose @('logs', '--tail', '20', 'gesture') }
            Show-Stream
        }
        'Logs'   { Invoke-Step 'Service log (Ctrl+C to stop following)' { Invoke-Compose @('logs', '-f', 'gesture') } }
        'Status' {
            Invoke-Step 'Container' { Invoke-Compose @('ps', '-a') }
            Invoke-Step 'Last log lines' { Invoke-Compose @('logs', '--tail', '10', 'gesture') }
            Show-Stream
        }
        'Down' { Invoke-Step 'Stopping and removing the container' { Invoke-Compose @('down') } }
        'Remove' {
            $answer = Read-Host "Delete the container, $Image and the NPU cache volume on $DeviceIp? (yes/no)"
            if ($answer -ne 'yes') { Write-Host 'Nothing removed.'; return }
            # The npu file only declares the cache volume here: 'down' creates nothing.
            Invoke-Step 'Removing the container and the cache volume' { Invoke-Compose @('down', '--volumes') -WithNpu }
            $ref = [uri]::EscapeDataString($Image)
            Invoke-Step "Removing $Image" { curl.exe -sS -f -X DELETE "$Api/images/$ref" }
        }
    }
} finally {
    Pop-Location
}
