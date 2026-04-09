# ztensor Development Log

## 2026-04-09: Issue #79 not reproducible at ztensor primitive level

**Type:** investigation
**Tags:** gpu, issue-79, patchtst, dgx-gb10

**Problem:** zerfoo PatchTST GPU training freezes at deterministic loss
0.268357 on DGX GB10. Issue #79 hypothesized the fault lies in ztensor's
GPU engine dst-output routing (`makeGPUResult` / `SetStorage` /
`GPUStorage.Slice()`). Four hypotheses (alpha/beta/gamma/delta) were
logged in the issue.

**Investigation:** Added `TestGPUEngine_PatchTSTBackward_DstRoundTrip`
(compute/gpu_dst_roundtrip_test.go) porting the exact op sequence from
`zerfoo/timeseries/patchtst_gpu_train.go:1022-1031`:
Transpose -> Zero -> MatMul(patchesT, dX, dPEW) -> in-place Add
accumulate into pre-seeded gradW -> gradW.Data() read. Ran on DGX GB10
via Spark pod `ztensor-issue79-repro-1775759440` (manifest at
`docs/bench/manifests/issue-79-repro.yaml`, commit 3e538e6 of
`fix/issue-79-matmul-accumulate-repro`).

Full test suite on DGX:
```
TestGPUEngine_Add_DstRoundTrip_OutOfPlace        PASS
TestGPUEngine_Add_DstRoundTrip_InPlace           PASS
TestGPUEngine_Add_DstRoundTrip_RepeatedInPlace   PASS
TestGPUEngine_Add_DstRoundTrip_NoExplicitSync    PASS
TestGPUEngine_PatchTSTBackward_DstRoundTrip      PASS
```

**Root cause:** Not in ztensor primitives. The
`Transpose -> Zero -> MatMul -> in-place Add` chain with a pre-seeded
CPU-wrapper dst does NOT reproduce zero readback on small shapes
(totalRows=4, patchLen=3, dModel=2). None of the four hypotheses from
the issue body is triggered at this level.

**Fix:** N/A. Investigation narrows the search to factors the ztensor
test does not exercise:
1. Shape regime -- production PatchTST uses thousands of rows / dModel in
   the hundreds; bug may only manifest under larger allocations or
   specific arena pressure.
2. Interaction with `encoderBackward` and multi-op state carried across
   the full batch, not just the patch-embedding backward slice.
3. The CPU-loop posEmb update at `patchtst_gpu_train.go:1012-1019`
   interleaved with GPU ops on the same stream.
4. zerfoo-side gradTs wrapper rebuild logic affecting how `.Data()`
   resolves after many accumulations.

**Impact:** Rules out ztensor engine primitive routing as the direct
cause of the frozen-loss signature. Next debugging must happen
zerfoo-side with a large-shape reproducer that closer matches the real
training configuration, or by instrumenting `trainWindowedGPU` itself
rather than trying to lift primitives into ztensor tests.

## 2026-03-29 -- v1.0.0 Benchmark Baseline

Pre-v1 benchmark baseline recorded on Apple M4 (darwin/arm64, 10 cores).

### tensor/

```
BenchmarkDequantizeAWQ-10        487695     2443 ns/op   6707.32 MB/s      0 B/op   0 allocs/op
BenchmarkDequantizeGPTQ-10       489553     2447 ns/op   6694.68 MB/s      0 B/op   0 allocs/op
BenchmarkDequantizeQ8-10         758720     1569 ns/op  10441.57 MB/s      0 B/op   0 allocs/op
BenchmarkDequantizeQ4-10         657984     1854 ns/op   8838.85 MB/s      0 B/op   0 allocs/op
BenchmarkQuantizeW8A8-10         202429     5773 ns/op   2838.19 MB/s   4928 B/op   2 allocs/op
BenchmarkDequantizeW8A8-10       750740     1606 ns/op  10201.41 MB/s      0 B/op   0 allocs/op
BenchmarkGemmW8A8-10                 16 66901216 ns/op               67125584 B/op   2 allocs/op
```

### compute/

```
BenchmarkCPUEngineMatMul/64x64x64-10              36901      32517 ns/op     16568 B/op    8 allocs/op
BenchmarkCPUEngineMatMul/128x128x128-10             6021     196675 ns/op     65720 B/op    8 allocs/op
BenchmarkCPUEngineAdd-10                            3769     371526 ns/op   4194458 B/op    6 allocs/op
BenchmarkCPUEngineMul-10                            3403     364828 ns/op   4194456 B/op    6 allocs/op
BenchmarkCPUEngineDiv-10                            3507     377398 ns/op   4194456 B/op    6 allocs/op
BenchmarkCPUEngineTranspose-10                       513    2160868 ns/op   4194593 B/op   10 allocs/op
BenchmarkCPUEngineSum-10                             988    1763374 ns/op      4456 B/op   10 allocs/op
BenchmarkBinaryOpSameShape-10                     184129       5682 ns/op      8344 B/op    6 allocs/op
BenchmarkBinaryOpBroadcast-10                      54372      25177 ns/op      8704 B/op   15 allocs/op
BenchmarkCPUEngineSoftmax-10                        1701     742775 ns/op   1048732 B/op    6 allocs/op
BenchmarkPowSquare-10                              83407      14530 ns/op      8448 B/op    8 allocs/op
BenchmarkPowGeneric-10                             17540      68821 ns/op      8696 B/op   15 allocs/op
BenchmarkMulScalarF32-10                          229659       5335 ns/op      8352 B/op    6 allocs/op
BenchmarkAddScalarF32-10                          257928       5093 ns/op      8352 B/op    6 allocs/op
BenchmarkDivScalarF32-10                          270468       4804 ns/op      8328 B/op    5 allocs/op
BenchmarkQ5KGEMVvsDequantReQuant-10                  117   10247756 ns/op     17868 B/op   30 allocs/op
BenchmarkQ6KGEMVvsDequantReQuant-10                  236    4881708 ns/op     17864 B/op   30 allocs/op
BenchmarkQ4KvsQ4_0GEMV/Q4_0-10                      1867     644002 ns/op     17864 B/op   30 allocs/op
BenchmarkQ4KvsQ4_0GEMV/Q4_K-10                       295    4109990 ns/op     18200 B/op   30 allocs/op
BenchmarkFusedRMSNorm/fused/1x128x1152-10          10000     101968 ns/op    590640 B/op   10 allocs/op
BenchmarkFusedRMSNorm/unfused/1x128x1152-10         1082    1082288 ns/op   3150078 B/op   83 allocs/op
BenchmarkFusedRMSNorm/fused/1x256x2048-10           3714     312384 ns/op   2098480 B/op   10 allocs/op
BenchmarkFusedRMSNorm/unfused/1x256x2048-10           472    2569119 ns/op   6298199 B/op  109 allocs/op
BenchmarkFusedRoPE/fused/1x128x256-10              41821      24126 ns/op    131240 B/op    6 allocs/op
BenchmarkFusedRoPE/unfused/1x128x256-10              838    1464438 ns/op   2370572 B/op  66139 allocs/op
BenchmarkFusedRoPE/fused/4x64x128-10              49312      24735 ns/op    131240 B/op    6 allocs/op
BenchmarkFusedRoPE/unfused/4x64x128-10               834    1434549 ns/op   2378861 B/op  66663 allocs/op
BenchmarkFusedSiLUGate/fused/1x128x1152-10         9127     128377 ns/op    590000 B/op    6 allocs/op
BenchmarkFusedSiLUGate/unfused/1x128x1152-10        1674     677999 ns/op   3146559 B/op   27 allocs/op
BenchmarkFusedSiLUGate/fused/1x256x2048-10          2968     366888 ns/op   2097328 B/op    6 allocs/op
BenchmarkFusedSiLUGate/unfused/1x256x2048-10          987    1228453 ns/op   6292712 B/op   40 allocs/op
BenchmarkMatMul_CPU_128-10                          5892     181247 ns/op     65720 B/op    8 allocs/op
BenchmarkMatMul_CPU_512-10                           140    8632963 ns/op   1048760 B/op    8 allocs/op
BenchmarkMatMul_CPU_1024-10                           16   66232242 ns/op   4194488 B/op    8 allocs/op
BenchmarkSoftmax_CPU-10                              469    2650127 ns/op  16777384 B/op    6 allocs/op
BenchmarkTensorPool_AcquireRelease-10              17619      65503 ns/op       176 B/op   13 allocs/op
BenchmarkTensorNew_Baseline-10                      3832     312565 ns/op   1048712 B/op    4 allocs/op
BenchmarkTensorArena_GetPut-10                  12423249        113.7 ns/op       0 B/op    0 allocs/op
BenchmarkTernaryGEMV/ternary_256x256-10            34976      34179 ns/op      1024 B/op    1 allocs/op
BenchmarkTernaryGEMV/dense_f32_256x256-10          24955      48088 ns/op      1024 B/op    1 allocs/op
BenchmarkTernaryGEMV/ternary_1024x1024-10           1506     796753 ns/op      4096 B/op    1 allocs/op
BenchmarkTernaryGEMV/dense_f32_1024x1024-10         1340     895470 ns/op      4096 B/op    1 allocs/op
BenchmarkTernaryGEMV/ternary_4096x4096-10             87   13621750 ns/op     16384 B/op    1 allocs/op
BenchmarkTernaryGEMV/dense_f32_4096x4096-10           81   14770621 ns/op     16384 B/op    1 allocs/op
```

### Key Observations

- **Fused ops deliver large speedups**: FusedRMSNorm is 8-10x faster than unfused; FusedRoPE is 58-60x faster; FusedSiLUGate is 3-5x faster.
- **Zero-alloc dequantization**: All dequantize paths (AWQ, GPTQ, Q8, Q4, W8A8) are allocation-free at 6.7-10.4 GB/s.
- **Tensor arena**: 113.7 ns/op with zero allocations for get/put cycle.
- **Ternary GEMV**: 1.1-1.4x faster than dense float32 GEMV at matching sizes.
