<#
.SYNOPSIS
    Build the arm64 NPU image on this Windows PC and deploy it to the OptixEdge test device over SSH.

.DESCRIPTION
    Automates README 7.7, the SSH variant of 7.9, 7.10 and 7.11. Works from any folder.

      Deploy  build + save the arm64 image, copy it with the compose files to the
              device, load it, start the stack, show log and status (default)
      Logs    follow the detector log on the device (Ctrl+C stops following)
      Status  list the containers and print the retained status message
      Check   run tools/npu_check.py once on the device (exit code 1 = not on the NPU)
      Test    run tests\test_integration.py from this PC against the device's broker
      Down    stop and remove the containers (image, cache volume and files are kept)
      Remove  Down + delete the image, the graph-cache volume and the remote folder

    -SkipBuild   do not build or copy the image (it is already on the device):
                 only refresh the compose files and run 'up'. Fast way to switch USE_NPU.
    -UseNpu      'true' (default) or 'false' = the negative control (README 7.10, point 6).
    -Engine      'podman' (default) or 'docker' (Docker Desktop) for the local build.

    The test device runs rootless Docker on a local socket and is reached over SSH.
    Without an SSH key, ssh/scp ask the password about three times per Deploy.
    To log in with a key instead (once):
      ssh-keygen -t ed25519
      type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh root@<ip> "cat >> ~/.ssh/authorized_keys"

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy_tools\deploy_device.ps1
.EXAMPLE
    .\deploy_tools\deploy_device.ps1 -SkipBuild -UseNpu false     # negative control
.EXAMPLE
    .\deploy_tools\deploy_device.ps1 -Action Check -DeviceIp 172.16.17.105
#>
param(
    [ValidateSet('Deploy', 'Logs', 'Status', 'Check', 'Test', 'Down', 'Remove')]
    [string]$Action = 'Deploy',
    [string]$DeviceIp = '172.16.17.105',
    [string]$User = 'root',
    [ValidateSet('podman', 'docker')]
    [string]$Engine = 'podman',
    [switch]$SkipBuild,
    [ValidateSet('true', 'false')]
    [string]$UseNpu = 'true',
    [string]$RemoteDir = '/persistent/anomaly_detection',   # '/' has only ~20 MB free
    [string]$RemoteDockerHost = 'unix:///run/user/985/docker.sock'   # rootless Docker
)

$ErrorActionPreference = 'Stop'

$Image   = 'localhost/anomaly_detection-detector:npu'
$Tar     = 'anomaly_detection-npu.tar'
$Target  = "$User@$DeviceIp"
$Compose = 'docker compose -p anomaly_detection -f docker-compose.yml -f docker-compose.npu.yml'
$ShowStatus = "$Compose exec -T mqtt-broker mosquitto_sub -v -t anomaly/paperclip/status -C 1 -W 15" +
              ' || echo WARNING: no status message yet'

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

# Runs shell commands on the device in one SSH session; stops at the first failure.
function Invoke-Remote([string]$Title, [string[]]$Commands, [switch]$Tty, [switch]$NoCd) {
    $prefix = @('set -e')
    if (-not $NoCd) { $prefix += "cd $RemoteDir" }
    $prefix += "export DOCKER_HOST=$RemoteDockerHost"
    $remoteScript = ($prefix + $Commands) -join '; '
    $sshArgs = @('-o', 'ConnectTimeout=10')
    if ($Tty) { $sshArgs += '-t' }
    Invoke-Step "$Title  [$Target]" { ssh @sshArgs $Target $remoteScript }
}

function Build-Image {
    if (-not (Get-Command $Engine -ErrorAction SilentlyContinue)) { throw "$Engine not found (README 7.2)." }
    $missing = 'autoencoder.tflite', 'autoencoder_int8.tflite', 'scaler.json', 'threshold.json', 'threshold_int8.json' |
        ForEach-Object { "models\$_" } | Where-Object { -not (Test-Path $_) }
    if ($missing) { throw "Missing $($missing -join ', '). Train the model first (README 7.4)." }

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
            podman build --platform linux/arm64 -f Dockerfile.npu -t $Image .
        }
    } else {
        if ($env:DOCKER_HOST) {
            throw "DOCKER_HOST is set to '$env:DOCKER_HOST': the build must run on this PC. Run 'Remove-Item Env:DOCKER_HOST' first."
        }
        if (-not (Test-Quiet { docker info })) { throw 'Docker is not running: start Docker Desktop.' }
        Invoke-Step 'Building the arm64 image (10-20 min the first time)' {
            docker buildx build --platform linux/arm64 -f Dockerfile.npu -t $Image --load .
        }
    }

    Remove-Item $Tar -ErrorAction SilentlyContinue
    Invoke-Step "Saving the image to $Tar (about 440 MB)" { & $Engine save -o $Tar $Image }
}

Push-Location (Split-Path $PSScriptRoot -Parent)   # the anomaly_detection folder
try {
    foreach ($tool in 'ssh', 'scp') {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "$tool not found. Windows: Settings > Optional features > OpenSSH Client."
        }
    }

    switch ($Action) {
        'Deploy' {
            $files = @('docker-compose.yml', 'docker-compose.npu.yml')
            $remote = @()
            if (-not $SkipBuild) {
                Build-Image
                $files += $Tar
                $remote += "docker load -i $Tar", "rm $Tar"
            }
            Invoke-Remote 'Creating the remote folder' @("mkdir -p $RemoteDir") -NoCd
            Invoke-Step "Copying $($files -join ', ') to ${Target}:$RemoteDir" {
                scp -o ConnectTimeout=10 @files "${Target}:$RemoteDir/"
            }
            $remote += "USE_NPU=$UseNpu $Compose up -d --no-build",
                       "$Compose ps",
                       'sleep 5',
                       "$Compose logs --tail 20 detector",
                       $ShowStatus
            Invoke-Remote "Starting the stack (USE_NPU=$UseNpu)" $remote
            Write-Host "`nDone. Expect accelerator=npu (or cpu with -UseNpu false). Follow the log with -Action Logs." -ForegroundColor Green
        }
        'Logs'   { Invoke-Remote 'Detector log (Ctrl+C to stop following)' @("$Compose logs -f detector") -Tty }
        'Status' { Invoke-Remote 'Containers and retained status' @("$Compose ps", $ShowStatus) }
        'Check' {
            Invoke-Remote 'One-shot NPU check (tools/npu_check.py)' @(
                "docker run --rm --device /dev/galcore -v /usr/lib:/host/usr/lib:ro $Image python tools/npu_check.py"
            ) -NoCd
        }
        'Test' {
            $python = if (Test-Path 'venv\Scripts\python.exe') { 'venv\Scripts\python.exe' } else { 'python' }
            if (-not (Test-Quiet { & $python -c 'import paho.mqtt' })) {
                throw "paho-mqtt not available to '$python'. Create the venv first (README 7.4)."
            }
            $previous = $env:MQTT_BROKER_HOST
            $env:MQTT_BROKER_HOST = $DeviceIp
            try {
                Invoke-Step "Integration test against ${DeviceIp}:1883" { & $python -X utf8 tests\test_integration.py }
            } finally {
                $env:MQTT_BROKER_HOST = $previous
            }
        }
        'Down' { Invoke-Remote 'Stopping and removing the containers' @("$Compose down") }
        'Remove' {
            $answer = Read-Host "Delete the stack, $Image, its graph-cache volume and $RemoteDir on $DeviceIp? (yes/no)"
            if ($answer -ne 'yes') { Write-Host 'Nothing removed.'; return }
            Invoke-Remote 'Removing everything this project created' @(
                "$Compose down || true",
                "docker rmi $Image || true",
                'docker volume rm anomaly_detection_npu-graph-cache || true',
                'cd /',
                "rm -rf $RemoteDir"
            )
        }
    }
} finally {
    Pop-Location
}
