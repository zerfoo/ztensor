#!/usr/bin/env bash
# GB10 runner for the engine-parity harness (ADR 091 harness #2, plan T1.2).
#
# Executed INSIDE the Spark pod (scripts/parity/parity-pod.yaml) on the DGX.
# Clones the requested ztensor ref, runs the GPU-gated parity tests
# (testing/parity, -run _GPU), and encodes correctness in the EXIT CODE
# (Spark drops stdout for completed pods): non-zero unless both GPU tests
# PASSED on real hardware. A CUDA-unavailable SKIP is a hard failure -- the
# whole point of this pod is real-GPU validation.
#
# Env (set in the pod manifest):
#   ZTENSOR_REF                git branch/tag/sha to test (default: main)
#   ZTENSOR_PARITY_REPORT_DIR where the per-schedule JSON reports land
#                              (mounted hostPath, default /reports)
set -euo pipefail

REF="${ZTENSOR_REF:-main}"
REPORT_DIR="${ZTENSOR_PARITY_REPORT_DIR:-/reports}"
export GOTOOLCHAIN=auto
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/zerfoo/lib:/usr/local/cuda/lib64}"
export ZTENSOR_PARITY_REPORT_DIR="$REPORT_DIR"

mkdir -p /work "$REPORT_DIR" && cd /work
git clone --depth 1 --branch "$REF" https://github.com/zerfoo/ztensor.git
cd ztensor
echo "HEAD: $(git rev-parse HEAD)"

set +e
go test ./testing/parity/ -run '_GPU$' -v -count=1 -timeout 600s > /tmp/out.txt 2>&1
code=$?
set -e
cat /tmp/out.txt
# Per-op PASS/FAIL lines for the streamed pod logs.
grep -- 'parity ' /tmp/out.txt || true

grep -q -- '--- PASS: TestParity_GPUvsCPU_ArenaStressSchedules_GPU' /tmp/out.txt || { echo FATAL: schedule parity not PASS; exit 3; }
grep -q -- '--- PASS: TestParity_GPURedProof_GPU' /tmp/out.txt || { echo FATAL: GPU red-proof not PASS; exit 5; }
grep -q -- '--- PASS: TestTrainingLoop_WolfPattern_GPU' /tmp/out.txt || { echo FATAL: Wolf-pattern training loop not PASS; exit 6; }
grep -q -- '--- SKIP: TestParity_GPUvsCPU_ArenaStressSchedules_GPU' /tmp/out.txt && { echo FATAL: SKIPPED no CUDA; exit 4; }
test "$code" -eq 0 || { echo "FATAL: go test exit $code"; exit "$code"; }

echo "reports:"
ls -l "$REPORT_DIR"
echo "VALIDATION_OK: CPU-vs-GPU parity (both schedules) + GPU red-proof + Wolf-pattern training loop passed on GB10"
