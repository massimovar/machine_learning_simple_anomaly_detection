<#
.SYNOPSIS
    Run the anomaly detector on this Windows PC with Docker Desktop (amd64, CPU).

.DESCRIPTION
    Automates README 7.5, 7.6 and 7.11. Works from any folder.

      Up      build the PC image, start broker + detector, show the status (default)
      Logs    follow the detector log (Ctrl+C stops following, not the container)
      Status  list the containers and print the retained status message
      Test    run tests\test_integration.py against the local broker
      Down    stop and remove the containers and their network (images are kept)

    Prerequisites: Docker Desktop running (it includes Compose) and a trained
    model in models\ (README 7.4).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy_tools\deploy_local_docker.ps1
.EXAMPLE
    .\deploy_tools\deploy_local_docker.ps1 -Action Test
#>
param(
    [ValidateSet('Up', 'Logs', 'Status', 'Test', 'Down')]
    [string]$Action = 'Up'
)

$ErrorActionPreference = 'Stop'

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

function Show-Status {
    Invoke-Step 'Containers' { docker compose ps }
    Write-Host "`n==> Retained status (anomaly/paperclip/status)" -ForegroundColor Cyan
    docker compose exec -T mqtt-broker mosquitto_sub -v -t anomaly/paperclip/status -C 1 -W 15
    if ($LASTEXITCODE -ne 0) { Write-Warning 'No status message yet: read the log with -Action Logs.' }
}

Push-Location (Split-Path $PSScriptRoot -Parent)   # the anomaly_detection folder
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'docker not found. Install Docker Desktop, or use deploy_local_podman.ps1.'
    }
    # A DOCKER_HOST left over from README 7.9 would send this PC stack to the device.
    if ($env:DOCKER_HOST) {
        throw "DOCKER_HOST is set to '$env:DOCKER_HOST'. This script targets this PC: run 'Remove-Item Env:DOCKER_HOST' first."
    }
    if (-not (Test-Quiet { docker info })) { throw 'Docker is not running: start Docker Desktop.' }
    if (-not (Test-Quiet { docker compose version })) { throw "'docker compose' not available: update Docker Desktop." }

    switch ($Action) {
        'Up' {
            $missing = 'models\autoencoder.tflite', 'models\scaler.json', 'models\threshold.json' |
                Where-Object { -not (Test-Path $_) }
            if ($missing) { throw "Missing $($missing -join ', '). Train the model first (README 7.4)." }

            Invoke-Step 'Building the PC image and starting broker + detector' { docker compose up -d --build }
            Show-Status
            Invoke-Step 'Detector log (last 15 lines)' { docker compose logs --tail 15 detector }
        }
        'Logs'   { Invoke-Step 'Detector log (Ctrl+C to stop following)' { docker compose logs -f detector } }
        'Status' { Show-Status }
        'Test' {
            $python = if (Test-Path 'venv\Scripts\python.exe') { 'venv\Scripts\python.exe' } else { 'python' }
            if (-not (Test-Quiet { & $python -c 'import paho.mqtt' })) {
                throw "paho-mqtt not available to '$python'. Create the venv first (README 7.4)."
            }
            $previous = $env:MQTT_BROKER_HOST
            $env:MQTT_BROKER_HOST = 'localhost'
            try {
                Invoke-Step 'Integration test against localhost:1883' { & $python -X utf8 tests\test_integration.py }
            } finally {
                $env:MQTT_BROKER_HOST = $previous
            }
        }
        'Down' { Invoke-Step 'Stopping and removing the containers' { docker compose down } }
    }
} finally {
    Pop-Location
}
