#!/usr/bin/env bash
# tune-host.sh — one-time host tuning for the NVIDIA DGX Spark (GB10).
#
# WHY vm.compaction_proactiveness=0:
#   On GB10 the GPU and the CPU share one physical pool. The kernel's background
#   page compactor migrates pages to build contiguous blocks, and every migrated
#   page is first unmapped from the GPU — that is the source of the periodic
#   4-5 s freezes (~10 % throughput) on this box. Setting it to 0 disables
#   background migration while direct compaction on allocation failure stays on.
#
# WHY the other two:
#   zone_reclaim_mode=0   never reclaim a remote zone to satisfy a local alloc
#   THP=never             avoids page-cache churn under large sequential reads
#
# All three are applied in one root shell, and the first is persisted in
# /etc/sysctl.d so it survives a reboot. The other two are runtime-only by
# design (THP is re-enabled at boot); persist them yourself if you want.
set -euo pipefail

now=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo "?")
echo "current: vm.compaction_proactiveness=${now}   (want 0)"

if [ "$now" = "0" ]; then
  echo "✓ already 0 — nothing to do"
  exit 0
fi

read -rp "Apply as root (sudo)? [y/N] " ans
[[ "${ans:-N}" =~ ^[Yy]$ ]] || { echo "aborted — nothing changed"; exit 0; }

sudo bash -c '
  printf "vm.compaction_proactiveness = 0\n" > /etc/sysctl.d/99-gb10-compaction.conf
  sysctl -q vm.compaction_proactiveness=0
  sysctl -q vm.zone_reclaim_mode=0
  echo never > /sys/kernel/mm/transparent_hugepage/enabled
  echo "  ✓ vm.compaction_proactiveness=$(cat /proc/sys/vm/compaction_proactiveness) (persisted)"
  echo "  ✓ vm.zone_reclaim_mode=$(cat /proc/sys/vm/zone_reclaim_mode)"
  echo "  ✓ transparent hugepage: $(cat /sys/kernel/mm/transparent_hugepage/enabled | tr -d "[]")"
'
echo "Done. Active now; only compaction_proactiveness persists across reboots."
