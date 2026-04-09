# ztensor Open GitHub Issues Resolution

## Context

Resolve all open GitHub issues in `github.com/zerfoo/ztensor`.

Status as of 2026-04-09:
- **#78** NCCL purego migration -- CLOSED via PR #80 (merged `af8af73`).
- **#79** GPU engine dst-output routing -- INVESTIGATION CLOSED ztensor-side.
  ztensor primitives cleared; follow-up must happen in zerfoo. Branch
  `fix/issue-79-matmul-accumulate-repro` retained as evidence.

No open ztensor issues remain assigned to this plan. If new issues are
filed, re-open this plan or start a fresh one.

## Evidence for #79 closure

Test file `compute/gpu_dst_roundtrip_test.go` on branch
`fix/issue-79-matmul-accumulate-repro` ports the exact backward-pass op
sequence from `zerfoo/timeseries/patchtst_gpu_train.go:1022-1031`:

```
Transpose(patches -> patchesT)
Zero(dPEW)
MatMul(patchesT, dX, dPEW)
Add(gradW, dPEW, gradW)                 # in-place accumulate
gradW.Data()
```

Ran 7 variants on DGX GB10 via Spark pod `ztensor-issue79-repro-1775761950`:

```
TestGPUEngine_Add_DstRoundTrip_OutOfPlace        PASS
TestGPUEngine_Add_DstRoundTrip_InPlace           PASS
TestGPUEngine_Add_DstRoundTrip_RepeatedInPlace   PASS
TestGPUEngine_Add_DstRoundTrip_NoExplicitSync    PASS
TestGPUEngine_PatchTSTBackward_DstRoundTrip      PASS  (4x3 / 4x2 tiny)
TestGPUEngine_PatchTSTBackward_RealisticShapes   PASS  (1600x8 / 1600x64, 20 iters)
TestGPUEngine_PatchTSTBackward_LargerBatch       PASS  (3200x8 / 3200x64, 20 iters)
```

None of the four hypotheses (alpha/beta/gamma/delta) from issue #79 is
triggered by the patch-embedding backward sequence at production shapes
and over many accumulation iterations. The `makeGPUResult` /
`SetStorage` / `GPUStorage.Slice()` path correctly routes dst tensors.

Findings posted to issue #79 as two comments; devlog entry dated
2026-04-09.

## Remaining suspects (zerfoo-side)

Future work, not tracked in this plan:
1. `encoderBackward` full op chain (dozens of ops per batch) -- not covered here.
2. CPU-loop `dPosData += dXData` interleave at `patchtst_gpu_train.go:1012-1019`
   forcing mid-backward D2H on the same stream.
3. zerfoo-side `gradTs` wrapper/arena state diverging from fresh `tensor.New` wrappers.

Recommended next action: instrument `trainWindowedGPU` directly
(log device pointers before/after each op on the first batch).

## Infra notes captured during investigation

- Spark silently drops `pod.spec.containers[0].command` when multi-element.
  Use `args: ["bash", "-c", ...]` with no `command` field.
- Spark silently truncates long `args[i]` strings. Put scripts on host at
  `/var/lib/zerfoo/bench-out/*.sh` and mount.
- Spark drops container stdout/stderr. Redirect to host file with
  `exec >...log 2>&1` inside the script.
- ztensor's `-tags cuda` build tag is unmaintained. The kernels package
  has only `//go:build !cuda` purego files. Default build is the GPU
  path. Do not pass `-tags cuda`.
- A prebuilt `/opt/zerfoo/lib/libkernels.so` exists on the DGX host and
  must be mounted into any pod running ztensor GPU tests.

Reference manifest: `docs/bench/manifests/issue-79-repro.yaml`.
Reference script: `/var/lib/zerfoo/bench-out/issue79-run.sh` on DGX host.

## Progress Log

### 2026-04-09 -- Investigation closed
- Merged PR #80, closed #78.
- Wrote, expanded, and ran 7-variant dst-routing test suite on DGX GB10.
- All variants PASS at production shapes with 20-iteration accumulation.
- Posted two update comments to issue #79 with findings.
- Trimmed plan: removed E3/E4/E5/E6 (diagnose/fix/release tracks) since
  the underlying premise -- reproducing #79 at ztensor level -- is
  disproven. Investigation continues in zerfoo if at all.
