#!/usr/bin/env bash
# Free the GB10's unified pool before a Cosmos run, and say whether it worked.
#
# There is ONE 119.7 GiB pool on this box shared by the GPU, the OS, the kernel's
# page cache and every pod. Two things eat it between runs:
#
#   1. rustfs, the Flyte object store, leaks anonymous heap. It climbs to tens of GB
#      and never gives it back. That memory is NOT reclaimable, so it is a straight
#      subtraction from what a model can have. A rollout restart reclaims it and the
#      data is safe on the PVC.
#   2. Page cache, from the checkpoint download or from staging it into the devbox.
#      This one is clean and reclaimable, which sounds harmless and is not: an
#      ORDINARY allocation triggers reclaim and succeeds, but creating a CUDA context
#      does not, and fails outright with CUDA_ERROR_OUT_OF_MEMORY from
#      cuDevicePrimaryCtxRetain while `free` cheerfully reports 110 GiB available.
#      That is a task dying in two seconds on a box that looks completely idle.
#
# Run this before `flyte run pipeline.py ...` on a box that has been up a while.

set -euo pipefail

# This script's own directory, so the reclaim helper below finds ./.venv whatever the
# caller's working directory is. It was previously referenced as $HERE without ever
# being set, which under `set -u` meant the fallback reclaim path -- the one that runs
# on every box without passwordless sudo, so in practice all of them -- aborted with
# "HERE: unbound variable" instead of freeing anything.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

RUSTFS_RESTART_GIB=${RUSTFS_RESTART_GIB:-6}

mem() {
    awk -v label="$1" '
        /^MemTotal:/     { total = $2 }
        /^MemAvailable:/ { avail = $2 }
        END { printf "%-8s available %6.1f GiB of %.1f GiB\n", label, avail/1048576, total/1048576 }
    ' /proc/meminfo
}

rustfs_gib() {
    # Sum RSS across every rustfs process; 0 if none are running.
    ps -o rss= -C rustfs 2>/dev/null | awk '{ s += $1 } END { printf "%.1f", s/1048576 }'
}

echo "== before =="
mem "host"
echo "rustfs   holding  $(rustfs_gib) GiB of anonymous heap"

held=$(rustfs_gib)
if awk -v h="$held" -v t="$RUSTFS_RESTART_GIB" 'BEGIN { exit !(h > t) }'; then
    echo
    echo "-- restarting rustfs (${held} GiB > ${RUSTFS_RESTART_GIB} GiB threshold) --"
    # Data lives on the PVC, so this only drops the leaked heap. Wait for the new
    # pod to be Ready before continuing: a Flyte task that starts while the object
    # store is down fails on its first blob write, which looks nothing like a memory
    # problem and wastes a debugging session.
    kubectl rollout restart deploy/rustfs -n flyte
    kubectl rollout status deploy/rustfs -n flyte --timeout=180s
else
    echo "rustfs under the ${RUSTFS_RESTART_GIB} GiB threshold, leaving it alone"
fi

echo
echo "-- dropping page cache (optional; needs passwordless sudo) --"
# -n only, never an interactive prompt: this script gets run from agents and CI where
# a password prompt is an invisible hang rather than a question.
if sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
    echo "dropped"
else
    # Not fatal, and usually not even worth doing. Page cache is reclaimable, so the
    # run gets the memory either way; dropping it only makes cuMemGetInfo stop
    # under-reporting. guard_memory() reads MemAvailable precisely so it is not fooled.
    echo "skipped (no passwordless sudo), falling back to forcing reclaim by hand"
    # NOT cosmetic, which is what this script used to claim. Page cache being
    # "reclaimable" is true for ordinary allocations and NOT true for the one that
    # matters here: creating a CUDA context. After a 33 GB copy left 107 GiB in cache
    # and 4 GiB genuinely free, every Cosmos task died on its very first CUDA call:
    #
    #   torch.AcceleratorError: CUDA error: out of memory
    #   Returning 2 (CUDA_ERROR_OUT_OF_MEMORY) from cuDevicePrimaryCtxRetain
    #
    # before a single weight was loaded. The driver does not push the kernel hard
    # enough to evict clean page cache, so the pages have to be taken from it by
    # someone who will: allocate and touch anonymous memory until MemFree is healthy,
    # then give it straight back. Nothing is destroyed, the cache is clean by
    # definition, and it needs no privileges at all.
    "$HERE/.venv/bin/python" - <<'RECLAIM' || echo "reclaim helper unavailable, continuing"
import gc, time

def meminfo():
    return {k: int(v.split()[0]) / 1048576.0
            for k, v in (line.split(":") for line in open("/proc/meminfo"))}

CHUNK, TARGET_FREE, FLOOR_AVAIL, CAP = 2, 85.0, 12.0, 100
before = meminfo()
print(f"  free {before['MemFree']:.1f} GiB, cache {before['Cached']:.1f} GiB")
blocks = []
try:
    while True:
        m = meminfo()
        if m["MemFree"] >= TARGET_FREE or m["MemAvailable"] < FLOOR_AVAIL:
            break
        if len(blocks) * CHUNK >= CAP:
            break
        b = bytearray(CHUNK * 1024**3)
        for off in range(0, len(b), 4096):   # touch every page so it is really committed
            b[off] = 1
        blocks.append(b)
finally:
    blocks.clear()
    gc.collect()
    time.sleep(2)
after = meminfo()
print(f"  reclaimed to free {after['MemFree']:.1f} GiB, cache {after['Cached']:.1f} GiB")
RECLAIM
fi

echo
echo "== after =="
mem "host"
echo "rustfs   holding  $(rustfs_gib) GiB of anonymous heap"

# The number that decides whether the run starts. guard_memory() applies the same
# 8 GiB headroom and refuses below ~46 GiB, so check it here rather than finding out
# after a 35 GB download.
awk '
    /^MemAvailable:/ { avail = $2/1048576 }
    END {
        budget = avail - 8
        printf "\nbudget after 8 GiB headroom: %.1f GiB\n", budget
        if (budget < 46) {
            printf "STILL TOO LOW for Cosmos3-Nano (~46 GiB). Something else is holding the pool:\n"
            printf "  ps aux --sort=-rss | head\n"
            exit 1
        }
        printf "enough for Cosmos3-Nano (~46 GiB). Good to run.\n"
    }
' /proc/meminfo
