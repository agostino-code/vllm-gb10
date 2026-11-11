#!/usr/bin/env bash
# tune-host.sh — Host optimization for NVIDIA DGX Spark / GB10
#
# Sets vm.compaction_proactiveness = 0 in /etc/sysctl.d/99-gb10-compaction.conf
#
# WHY: On DGX Spark (GB10 unified memory), the kernel's background page compactor
# migrates pages to build large contiguous blocks. Because GPU and CPU share the
# same physical memory pool, every migrated page is first unmapped from the GPU.
# This causes periodic 4-5 second freezes every ~37 seconds (~10% throughput loss).
# Setting vm.compaction_proactiveness = 0 disables this background migration,
# eliminating the freezes while direct compaction on real allocation failure remains active.
set -euo pipefail

CMD='printf "vm.compaction_proactiveness = 0\n" > /etc/sysctl.d/99-gb10-compaction.conf && sysctl -q vm.compaction_proactiveness=0 && echo "  ✓ vm.compaction_proactiveness=$(cat /proc/sys/vm/compaction_proactiveness) (persisted)"'
now=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo "?")
echo "current: vm.compaction_proactiveness=${now}   (want 0)"
if [ "$now" = "0" ]; then
  echo "✓ already set to 0 — nothing to do"
  exit 0
fi

echo
echo "This will run AS ROOT (sudo):"
echo "  sudo bash -c '$CMD'"
echo
read -rp "Proceed? [y/N] " ans
if [[ ! "${ans:-N}" =~ ^[Yy]$ ]]; then
  echo "aborted — nothing changed"
  exit 0
fi

sudo bash -c "$CMD"
echo "Done. Active now (no restart needed) and persists across reboots."
