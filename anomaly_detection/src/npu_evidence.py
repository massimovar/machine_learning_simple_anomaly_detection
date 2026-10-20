"""
npu_evidence.py — facts about the NPU that do not come from our own code
=========================================================================

WHY THIS FILE EXISTS:
    "accelerator": "npu" in our status is a self-report: our program saying
    what it believes. Stronger proof comes from things other layers leave
    behind:

    graph cache   With VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1 the VeriSilicon
                  driver stores the graph it compiled for the NPU as a .nb
                  file. Only an NPU compilation produces that file.
    galcore       The NPU driver's version: the kernel driver's when readable,
                  else the one compiled into the lent user-space driver (libGAL).

Every function returns None or an empty result instead of raising: missing
evidence is information, not an error.
"""

import os
import re

# "$VERSION$6.4.11:745085:cb782f4af5$" -> "6.4.11:745085" (release:build, without the git hash)
_GAL_VERSION = re.compile(rb"\$VERSION\$(\d+\.\d+\.\d+[^:$]*(?::\d+)?)")


def graph_cache(environ=None) -> dict:
    environ = os.environ if environ is None else environ
    enabled = environ.get("VIV_VX_ENABLE_CACHE_GRAPH_BINARY") == "1"
    directory = environ.get("VIV_VX_CACHE_BINARY_GRAPH_DIR")
    info = {"enabled": enabled, "dir": directory, "files": 0, "bytes": 0, "names": []}
    if not (enabled and directory and os.path.isdir(directory)):
        return info
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if name.endswith(".nb") and os.path.isfile(path):
            info["files"] += 1
            info["bytes"] += os.path.getsize(path)
            info["names"].append(name)
    info["names"] = info["names"][:5]
    return info


def galcore_version(paths=("/sys/module/galcore/version", "/sys/kernel/debug/gc/version"),
                    gal_library=None):
    """The NPU driver's version, from the best source this process can read.

    WHY two kinds of source: the kernel driver's own version file is in sysfs
    on some BSPs and only in debugfs (root-only) on others, e.g. the OptixEdge.
    A rootless container then falls back to the version string compiled into
    the lent user-space driver (libGAL), which NXP ships with the kernel driver.
    The prefix says which one was read.
    """
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                words = f.readline().split()
            if words:
                return f"galcore {words[0]}"
        except OSError:
            pass
    if gal_library is None:
        gal_library = os.path.join(os.environ.get("HOST_LIB_DIR", "/host/usr/lib"), "libGAL.so")
    try:
        with open(gal_library, "rb") as f:
            match = _GAL_VERSION.search(f.read())
    except OSError:
        return None
    return f"libGAL {match.group(1).decode('ascii')}" if match else None
