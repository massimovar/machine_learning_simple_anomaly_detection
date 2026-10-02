#!/bin/sh
# ============================================================================
# npu_probe.sh — collect the facts needed to enable the NPU on the device
# ============================================================================
# WHY THIS FILE EXISTS:
#   The i.MX 8M Plus NPU is driven by the "galcore" kernel driver.  Its
#   user-space libraries (VX delegate, TIM-VX, OpenVX, GAL) must come from
#   the same NXP BSP release as that driver.  Before choosing how the gesture
#   container gets the NPU (host libraries mounted in, an image built from
#   the BSP, or CPU only), we need to know what the device has.
#
#   The script only READS information: it installs and changes nothing.
#
# RUN on the device (as root for complete output) and keep the report:
#   sh npu_probe.sh > npu_probe.txt 2>&1
#   sh npu_probe.sh <image> > npu_probe.txt 2>&1
#       also checks that a container started from <image> sees /dev/galcore
# Then measure the model: tools/bench_pose.py (with and without --npu).
# ============================================================================

# show "<command>": print the command, then its output
show() {
    printf '$ %s\n' "$1"
    sh -c "$1" 2>&1 || echo "(not available)"
    echo
}

echo "=== System ==="
show "uname -a"
show "cat /etc/os-release"
show "tr -d '\\000' < /proc/device-tree/model; echo"
show "cat /sys/devices/soc0/soc_id /sys/devices/soc0/revision"
show "nproc"
show "grep -E 'MemTotal|MemAvailable' /proc/meminfo"

echo "=== NPU kernel driver (galcore) ==="
show "ls -l /dev/galcore"
show "cat /sys/module/galcore/version"
show "dmesg | grep -i galcore | tail -n 5"

echo "=== NPU user-space libraries (must come from the same BSP as the driver) ==="
for lib in libvx_delegate libtim-vx libOpenVX libOpenVXU libGAL libVSC libArchModelSw libNNArchPerf; do
    found=$(find /usr/lib /usr/lib64 /lib -maxdepth 2 -name "$lib.so*" 2>/dev/null)
    if [ -n "$found" ]; then
        ls -l $found
    else
        echo "$lib: not found"
    fi
done
echo

echo "=== TensorFlow Lite on the device ==="
show "ls -d /usr/bin/tensorflow-lite-*"
show "python3 -c 'import tflite_runtime; print(tflite_runtime.__version__)'"

echo "=== C runtime (container must be compatible to load host NPU libraries) ==="
show "ldd --version | head -n 1"
show "ls -l /usr/lib/libstdc++.so.6*"

echo "=== Container runtime ==="
show "docker version --format 'client {{.Client.Version}}, server {{.Server.Version}}'"
show "docker info | grep -E 'Server Version|Storage Driver|Cgroup|Runtimes|Security Options|Architecture'"
if [ -n "${1:-}" ]; then
    show "docker run --rm --device /dev/galcore --entrypoint ls '$1' -l /dev/galcore"
fi

echo "=== Temperatures (millidegrees C) ==="
for zone in /sys/class/thermal/thermal_zone*; do
    [ -r "$zone/temp" ] && echo "$(cat "$zone/type"): $(cat "$zone/temp")"
done
