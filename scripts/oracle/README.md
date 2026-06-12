# PyTorch-as-oracle per-op parity harness (ADR 091, T1.3)

Runs the same op with the same inputs through ztensor and through PyTorch
(`nvcr.io/nvidia/pytorch:26.02-py3` on the DGX GB10) and diffs forward AND
backward outputs within per-op tolerances. Catches numerics-convention
divergence -- fast-math approximations, reduction ordering, eps placement --
that ztensor's CPU and GPU engines could share, which gradcheck cannot see.

This is **test infrastructure only**: the production stack stays pure Go. The
torch dependency is confined to `run_oracle.py` inside the pinned NGC
container.

## Pieces

| Piece | Where | Role |
|---|---|---|
| Case-bundle writer | `testing/oracle` (Go) | runs each gradcheck-registry op fwd+bwd, dumps inputs/upstream/outputs/grads |
| Bundle generator CLI | `testing/oracle/cmd/oracle-gen` | writes one bundle per op + `generation.json` summary |
| Op -> torch mapping | `testing/oracle/torchmap.go` | torch expression per op; SKIP list with reasons |
| Python runner | `scripts/oracle/run_oracle.py` | replays bundles in torch, emits `report.json` diffs |
| Spark pod template | `scripts/oracle/oracle-pod.yaml` | runs the runner on the GB10 with a GPU |
| Red proof (CI) | `testing/oracle/redproof_test.go` | fast-math-tanh fixture must FAIL the diff |

## Bundle format (format_version 1)

One directory per op: `manifest.json` + raw little-endian row-major tensor
files (no header). The manifest records the op name, the torch expression,
shapes, dtypes (`float32`/`float64`, numpy names), the generation seed, and
per-op tolerances. Authoritative spec: `testing/oracle/bundle.go`.

```
bundles/
  generation.json          # written/skipped summary
  Tanh/
    manifest.json
    input_0.bin            # op inputs (x0, x1, ...)
    upstream.bin           # upstream gradient dL/dy fed to Backward
    forward.bin            # ztensor forward output
    grad_input_0.bin       # ztensor input gradients
  LayerNorm/
    ...
    param_gamma.bin        # trainable parameter values
    grad_param_gamma.bin   # ztensor parameter gradients
```

Pass criterion per element (identical in `diff.go` and `run_oracle.py`):
`|ztensor - torch| <= atol + rtol * |torch|`; any NaN fails.

## DGX run procedure (all GPU work via Spark, serialized)

```sh
SPARK=http://192.168.86.250:8080
RUNID=$(git rev-parse --short=8 HEAD)

# 1. Generate bundles (dev machine, CPU f32 first cut).
go run github.com/zerfoo/ztensor/testing/oracle/cmd/oracle-gen \
  -out /tmp/oracle/$RUNID/bundles

# 2. Stage bundles + runner on the DGX (file transfer only -- workloads
#    themselves must go through Spark).
rsync -av /tmp/oracle/$RUNID/bundles scripts/oracle/run_oracle.py \
  ndungu@192.168.86.250:/home/ndungu/oracle/$RUNID/

# 3. Confirm the GPU is free, then submit the pod (replace RUNID in the
#    template; Spark rejects duplicate pod names).
curl -s $SPARK/api/v1/resources
sed "s/RUNID/$RUNID/g" scripts/oracle/oracle-pod.yaml | \
  curl -sf -X POST $SPARK/api/v1/pods \
    -H 'Content-Type: application/yaml' --data-binary @-

# 4. Poll until Succeeded/Failed; tail logs for the per-op PASS/FAIL lines.
curl -s $SPARK/api/v1/pods/ztensor-oracle-$RUNID
curl -s $SPARK/api/v1/pods/ztensor-oracle-$RUNID/logs

# 5. Collect the report and clean up.
rsync -av ndungu@192.168.86.250:/home/ndungu/oracle/$RUNID/report.json \
  /tmp/oracle/$RUNID/
curl -s -X DELETE $SPARK/api/v1/pods/ztensor-oracle-$RUNID
```

The runner exits 0 only if every bundle passed; the pod phase reflects that.
`report.json` carries per-op `max_abs`/`max_rel` for forward and for every
input/parameter gradient -- attach it to the devlog entry for the run.

### GPU-engine bundles (`-engine gpu`)

`oracle-gen -engine gpu` records the **CUDA GPU engine** through the exact
same bundle format: a Go runner on the DGX (Spark pod, golang image + mounted
CUDA + `libkernels.so`) regenerates the bundles with `compute.GPUEngine`
outputs, then the same `run_oracle.py` invocation judges them. That is the
gate for the kernel numerics work (plan T3.x: fast-math removal, reduction
ordering) -- the recorded forward/gradients come from the CUDA kernels under
test. Generation MUST run on the GB10 via Spark (it needs CUDA); point
`LD_LIBRARY_PATH` at the kernel build under test (e.g. a fresh
`/home/ndungu/ztensor-kernels-build-<sha>` hostPath) so the bundles record
that exact `libkernels.so`.

## Skipped ops

| Op | Reason |
|---|---|
| HadamardTransform | torch has no built-in normalized Walsh-Hadamard transform; hand-building the H matrix in the runner would test our own reimplementation rather than torch |

## Red proof (CI-side)

`testing/oracle/redproof_test.go` encodes the historical ztensor#125 bug
class without torch: a Tanh bundle whose recorded "ztensor output" comes from
an unsaturated fast-math-style tanh (grows past `|x| > 9` instead of clamping
to +-1). The Go reference checker -- the same diff logic as `run_oracle.py`,
with `math.Tanh` standing in for `torch.tanh` -- must flag it red, and the
correct-tanh twin must pass green. This proves the diff/report semantics in
CI; the real torch comparison runs on the GB10.
