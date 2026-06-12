# Engine parity under arena stress (ADR 091 harness #2, T1.2)

Runs the same op set (the gradcheck OpInfo registry) through the **CPU f32
engine and the GPU f32 engine**, forward AND backward, in **interleaved
arena-stress schedules**, and diffs forward outputs, input gradients, and
parameter gradients per op within per-op tolerances. This is the harness
shape that catches kernel bugs and the cached-intermediate corruption class
(zerfoo#842 LayerNorm variance, Wolf QK-norm) that single-op tests cannot
see.

## Schedules

```
phase F:  opA.Forward, opB.Forward, ..., opZ.Forward
          [reset point]
phase B:  opZ.Backward, ..., opB.Backward, opA.Backward
```

| Variant | Reset point | What it proves |
|---|---|---|
| `no-reset` | nothing | pure kernel parity (no allocator interference) |
| `reset-between-fwd-bwd` | candidate arena `Reset()` | the Wolf per-sample-ResetPool hazard: ops honoring the save-for-backward contract (ztensor ADR 006) or recomputing from live inputs stay correct; raw struct-field caches read recycled memory and are flagged |

The GPU engine is built with a **deliberately small arena** (64 MiB via
`compute.SetArenaBytesForTesting`, far below the 1 GB env-var minimum) and
**poison-on-reset** (`ZTENSOR_ARENA_POISON` semantics), so any read of a
recycled buffer is a deterministic NaN attributed to the op + schedule in
the report, not a delayed training NaN days later.

## Tolerances

Defaults are f32-appropriate: forward `atol 1e-6 / rtol 1e-5`, gradients
`rtol 1e-4`. MatMul, Softmax, LayerNorm, reductions, and Hadamard are looser
(`rtol` up to `1e-3` on gradients): CPU loops and cuBLAS/parallel kernels
legitimately reduce in different orders at f32, and that expected divergence
is exactly why per-op tolerances exist. Per-op overrides:
`testing/parity/parity.go` (`toleranceOverrides`, `Op.Tol`).

## CI vs GB10 split

| Piece | Where it runs |
|---|---|
| Schedule runner + diff logic + report (`testing/parity`) | ordinary CI |
| Host-backed-arena candidate (`StressEngine`): CPU engine vs CPU-engine-with-arena under both schedules, poison on | ordinary CI -- catches lifetime bugs without a GPU |
| Red proofs: cached-intermediate fixture pair + contract-stripped real op | ordinary CI (and re-proven on GPU) |
| CPU-f32 vs GPU-f32 over the full registry, both schedules | GB10 only (`-run _GPU`, skips without CUDA) |

The CI candidate (`StressEngine`) relocates every engine result into a
host-backed `cuda.ArenaPool`, giving CPU tensors GPU lifetime semantics --
the same trick as the `TestSaveForBackward_WolfHazard_*` tests.

## Red proof

`testing/parity/fixture.go` ships a fixture pair encoding the pre-fix
LayerNorm bug shape:

- `FixtureCachedIntermediateRaw` caches an engine-computed forward
  intermediate in a struct field with **no** SaveForBackward. Under
  `reset-between-fwd-bwd` + poison the harness MUST flag it (NaN diff,
  `max_abs = +Inf`, attributed to the op and schedule). Asserted red in CI
  and on the GB10.
- `FixtureCachedIntermediateContract` registers the same intermediate via
  the contract and passes green everywhere.

`TestRun_RedProof_RealOpWithoutContractFlagged` additionally strips the
contract from a real registry op (Exp) and asserts the harness flags it.

## Report

One JSON per (candidate, schedule), oracle-report style
(`run_oracle.py` conventions): per-op `max_abs`/`max_rel` for forward and
every gradient, `passed`/`failed`/`errored` totals. Non-finite diffs encode
as `"Infinity"`/`"NaN"` strings. Attach the GB10 reports to the devlog entry
for the run.

## DGX run procedure (all GPU work via Spark, serialized)

```sh
SPARK=http://192.168.86.250:8080
RUNID=$(git rev-parse --short=8 HEAD)

# 1. Report directory on the DGX (file staging only -- workloads go
#    through Spark).
ssh ndungu@192.168.86.250 mkdir -p /home/ndungu/parity/$RUNID

# 2. Render the pod: embed run.sh (base64, single line -- Spark v1.13.1
#    mangles YAML block scalars) and the run id.
B64=$(base64 -i scripts/parity/run.sh | tr -d '\n')
sed -e "s/RUNID/$RUNID/g" -e "s|<BASE64>|$B64|" scripts/parity/parity-pod.yaml > /tmp/parity-pod.yaml

# 3. Confirm the GPU is free, then submit.
curl -s $SPARK/api/v1/resources
curl -sf -X POST $SPARK/api/v1/pods \
  -H 'Content-Type: application/yaml' --data-binary @/tmp/parity-pod.yaml

# 4. Poll until Succeeded/Failed; tail logs for the per-op `parity PASS/FAIL`
#    lines and the final VALIDATION_OK.
curl -s $SPARK/api/v1/pods/ztensor-parity-$RUNID
curl -s $SPARK/api/v1/pods/ztensor-parity-$RUNID/logs

# 5. Collect the reports and clean up.
rsync -av ndungu@192.168.86.250:/home/ndungu/parity/$RUNID/ /tmp/parity/$RUNID/
curl -s -X DELETE $SPARK/api/v1/pods/ztensor-parity-$RUNID
```

`run.sh` exits non-zero unless BOTH GPU tests genuinely passed (a
CUDA-unavailable SKIP is a hard failure), so the pod phase is the verdict.
Test a non-main ref by setting `ZTENSOR_REF` in the manifest env.

## Wolf-pattern training loop (S2.3.1)

`TestTrainingLoop_WolfPattern_GPU` adds the third GB10 gate: a small
two-layer training loop running the exact Wolf gr-12 hazard schedule
(per-sample forward+backward, gradients accumulated in place into
persistent non-arena device buffers, `ResetPool` after every sample,
in-place SGD step once per batch) under poison with the small arena.
PASS requires finite parameters AND f32-tolerance agreement with a plain
CPU-engine run of the byte-identical loop. A CI twin
(`TestTrainingLoop_WolfPattern_StressCI`) runs the same loop against the
host-backed-arena StressEngine, so lifetime regressions in the pattern
are caught before a GB10 round trip.
