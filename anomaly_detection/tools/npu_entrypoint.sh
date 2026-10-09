#!/bin/sh
# ============================================================================
# npu_entrypoint.sh — borrow the host's NPU libraries, then start the command
# ============================================================================
# WHY THIS FILE EXISTS:
#   The NPU needs three layers that must match each other:
#     kernel driver (galcore)       -> always the host's: containers share the kernel
#     NPU user-space libraries      -> OpenVX/GAL/TIM-VX/VX delegate, built for that driver
#     TFLite runtime + Python app   -> the host's tflite_runtime (borrowed read-only,
#                                      same libtensorflow-lite as the delegate), else
#                                      the image's ai-edge-litert
#   We install NOTHING on the host.  The host's /usr/lib is mounted READ-ONLY
#   at /host/usr/lib, and this script links ONLY the NPU libraries listed in
#   NPU_HOST_LIBS into /opt/npu/lib, which goes first on LD_LIBRARY_PATH.
#
#   WHY not simply LD_LIBRARY_PATH=/host/usr/lib?  That folder also holds the
#   host's libc, libstdc++, libpython...  The loader would mix them with the
#   image's own copies and the process would crash in confusing ways.
#   Borrow the minimum; the image provides everything else.
#
#   The script then runs `ldd` on the delegate: any "not found" line names
#   exactly what is still missing, before Python even starts.
#
# USAGE (Dockerfile.npu sets it as ENTRYPOINT):
#   docker run --device /dev/galcore -v /usr/lib:/host/usr/lib:ro IMAGE [command]
# Env:
#   HOST_LIB_DIR     where the host libs are mounted   (default /host/usr/lib)
#   NPU_HOST_LIBS    space-separated glob patterns to borrow
#   VX_DELEGATE_PATH delegate to load (default: the borrowed libvx_delegate.so)
#   NPU_DEBUG=1      print the full ldd resolution (shows which file each lib comes from)
#   HOST_TFLITE_RUNTIME=0  do not borrow the host's tflite_runtime (use ai-edge-litert)
# ============================================================================
set -eu

HOST_LIB_DIR="${HOST_LIB_DIR:-/host/usr/lib}"
NPU_LIB_DIR="${NPU_LIB_DIR:-/opt/npu/lib}"
# i.MX 8M Plus NPU stack (NXP BSP): delegate -> TIM-VX -> OpenVX -> GAL -> /dev/galcore.
# libNN*/libOvx*/libCLC hold pre-compiled NPU kernels and the shader compiler.
# libtensorflow-lite.so*: the device's delegate is dynamically linked against
# the host's TFLite shared library (ldd shows libtensorflow-lite.so.2.16.2).
NPU_HOST_LIBS="${NPU_HOST_LIBS:-libvx_delegate.so* libtim-vx.so* libOpenVX.so* libOpenVXU.so* libGAL.so* libVSC.so* libArchModelSw.so* libNNArchPerf.so* libNNVXCBinary*.so* libNNGPUBinary*.so* libOvx12VXCBinary*.so* libOvxGPUVXCBinary*.so* libCLC.so* libtensorflow-lite.so*}"

if [ -d "$HOST_LIB_DIR" ]; then
    mkdir -p "$NPU_LIB_DIR"
    linked=0
    for pattern in $NPU_HOST_LIBS; do
        for lib in "$HOST_LIB_DIR"/$pattern; do
            [ -e "$lib" ] || continue
            ln -sf "$lib" "$NPU_LIB_DIR/$(basename "$lib")"
            linked=$((linked + 1))
        done
    done
    echo "[npu] linked $linked host libraries from $HOST_LIB_DIR into $NPU_LIB_DIR"
    export LD_LIBRARY_PATH="$NPU_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export VX_DELEGATE_PATH="${VX_DELEGATE_PATH:-$NPU_LIB_DIR/libvx_delegate.so}"

    if [ -e "$VX_DELEGATE_PATH" ]; then
        resolution=$(ldd "$VX_DELEGATE_PATH" 2>&1 || true)
        missing=$(printf '%s\n' "$resolution" | grep "not found" || true)
        if [ -n "$missing" ]; then
            echo "[npu] WARNING: the delegate needs libraries that are not available:"
            printf '%s\n' "$missing"
            echo "[npu] add their names to NPU_HOST_LIBS (host lib) or install them in the image"
        else
            echo "[npu] every dependency of $VX_DELEGATE_PATH resolves"
        fi
        if [ "${NPU_DEBUG:-0}" = "1" ]; then
            printf '%s\n' "$resolution"
        fi
    else
        echo "[npu] no delegate at $VX_DELEGATE_PATH: the detector will run on the CPU"
    fi

    # Borrow the host's tflite_runtime Python package too (read-only, nothing
    # installed).  WHY: the host's delegate is linked against the host's
    # libtensorflow-lite.so; tflite_runtime uses that SAME library, so delegate
    # and interpreter share one TFLite copy (the pairing NXP builds and tests).
    # It is a compiled CPython module, so the image's Python version must equal
    # the host's (python3.12 on our device).  HOST_TFLITE_RUNTIME=0 skips it and
    # the image's own ai-edge-litert is used instead.
    if [ "${HOST_TFLITE_RUNTIME:-1}" = "1" ]; then
        pyver=$(python -c 'import sys; print("python%d.%d" % sys.version_info[:2])')
        host_pkg="$HOST_LIB_DIR/$pyver/site-packages/tflite_runtime"
        if [ -d "$host_pkg" ]; then
            mkdir -p "$NPU_LIB_DIR/python"
            ln -sfn "$host_pkg" "$NPU_LIB_DIR/python/tflite_runtime"
            export PYTHONPATH="$NPU_LIB_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
            echo "[npu] using the host's tflite_runtime ($host_pkg)"
        else
            echo "[npu] no host tflite_runtime for $pyver: using the image's runtime"
        fi
    fi
else
    echo "[npu] $HOST_LIB_DIR is not mounted: NPU libraries unavailable, CPU only"
fi

if [ -e /dev/galcore ]; then
    echo "[npu] /dev/galcore is present"
else
    echo "[npu] /dev/galcore is missing (run with --device /dev/galcore): CPU only"
fi

exec "$@"
