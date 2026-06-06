#!/usr/bin/env bash
# Out-of-band uninterruptible-sleep (D-state) watchdog for the GB10 bulk-upload
# wedge (zerfoo/ztensor#106).
#
# When a CUDA driver ioctl wedges in D-state, every in-container capture path
# that touches the GPU (exec, logs, ssh) hangs -- but a plain process that was
# already running and only writes to a file keeps going (like Wolf's heartbeat
# goroutine). This script is that process: started BEFORE the wedge, in the same
# container/PID namespace as the workload, it samples every D-state thread's
# /proc/<tid>/{wchan,syscall,stack,status} on a loop and appends to a file on a
# hostPath that survives the data-plane wedge. A separate (non-wedged) reader pod
# mounting the same hostPath exfiltrates the captured frames.
#
# Env:
#   WEDGE_WATCH_OUT       output file (default /work/wedge-out/watchdog.log)
#   WEDGE_WATCH_INTERVAL  seconds between samples (default 2)
#   WEDGE_WATCH_COMM      if set, only report threads whose process comm matches
set -u
OUT="${WEDGE_WATCH_OUT:-/work/wedge-out/watchdog.log}"
INTERVAL="${WEDGE_WATCH_INTERVAL:-2}"
WANT_COMM="${WEDGE_WATCH_COMM:-}"
mkdir -p "$(dirname "$OUT")"

{
  echo "=== dstate-watchdog start $(date -u +%FT%TZ) interval=${INTERVAL}s comm-filter='${WANT_COMM}' ==="
  echo "kptr_restrict=$(cat /proc/sys/kernel/kptr_restrict 2>/dev/null) uid=$(id -u)"
  while true; do
    found=0
    for taskdir in /proc/[0-9]*/task/[0-9]*; do
      stat=$(cat "$taskdir/stat" 2>/dev/null) || continue
      # Field 3 is the state, but comm (field 2) is parenthesized and may contain
      # spaces/parens -- take everything after the last ')' then the first token.
      state=$(printf '%s' "$stat" | sed 's/.*)[[:space:]]*//' | awk '{print $1}')
      [ "$state" = "D" ] || continue
      tid=$(basename "$taskdir")
      pid=$(printf '%s' "$taskdir" | cut -d/ -f3)
      pcomm=$(cat "/proc/$pid/comm" 2>/dev/null)
      if [ -n "$WANT_COMM" ] && [ "$pcomm" != "$WANT_COMM" ]; then continue; fi
      tcomm=$(cat "$taskdir/comm" 2>/dev/null)
      echo "--- $(date -u +%FT%TZ) D-STATE pid=$pid tid=$tid pcomm=$pcomm tcomm=$tcomm ---"
      echo "wchan:   $(cat "$taskdir/wchan" 2>/dev/null)"
      echo "syscall: $(cat "$taskdir/syscall" 2>/dev/null)   # arm64 ioctl=29; fields: nr arg0(fd) arg1(req) ..."
      echo "stack:"
      cat "$taskdir/stack" 2>/dev/null | sed 's/^/  /' || echo "  (stack unavailable: needs root + CONFIG_STACKTRACE + kptr_restrict<2)"
      grep -E '^(Name|State|VmRSS):' "$taskdir/status" 2>/dev/null | sed 's/^/  /'
      found=1
    done
    if [ "$found" = 1 ]; then echo "=== sample end $(date -u +%FT%TZ) ==="; fi
    sleep "$INTERVAL"
  done
} >> "$OUT" 2>&1
