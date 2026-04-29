# Changelog

## [1.8.0](https://github.com/zerfoo/ztensor/compare/v1.7.0...v1.8.0) (2026-04-29)


### Features

* **compute:** bulk-upload F32 weights to one device buffer ([#103](https://github.com/zerfoo/ztensor/issues/103)) ([9ca83f6](https://github.com/zerfoo/ztensor/commit/9ca83f6a234ba345a2ead3417b8c98d584fa5d03))

## [1.7.0](https://github.com/zerfoo/ztensor/compare/v1.6.0...v1.7.0) (2026-04-20)


### Features

* **compute:** add StepScope + MarkStepBoundary for training-loop arena reset ([6356dde](https://github.com/zerfoo/ztensor/commit/6356dde3b46ad290ad1f3e0baffe28386bc8c5ef))
* **compute:** configurable GPU arena size via ZERFOO_ARENA_SIZE_GB ([cab6067](https://github.com/zerfoo/ztensor/commit/cab606733052415b6ba771d4bf839f0116d0ab37))

## [1.6.0](https://github.com/zerfoo/ztensor/compare/v1.5.0...v1.6.0) (2026-04-17)


### Features

* **compute:** T1.2 add ensureNotCapturing guard and ErrCaptureIncompatibleAllocation ([18e1f5a](https://github.com/zerfoo/ztensor/commit/18e1f5a0a7d5c8b5ae0de50071be5614002ada4a))
* **compute:** T2.1a add WithCapture helper for capture-aware graph lifecycle ([d60c902](https://github.com/zerfoo/ztensor/commit/d60c90294fee387614d292317c1683a08ee52461))
* **compute:** T2.2 capture-aware allocWeight routing via cudaMallocAsync ([2a723b7](https://github.com/zerfoo/ztensor/commit/2a723b7ac4684d1c28466924a2a4eb9029007ca0))
* **compute:** T2.3 pre-allocate workspace buffers at UploadWeights to avoid capture-time alloc ([9f9eb5c](https://github.com/zerfoo/ztensor/commit/9f9eb5c9f7960ead4046e01e0fb54bb5f87f7888))
* **cuda:** T1.1 add StreamCaptureStatus purego binding ([879cbc9](https://github.com/zerfoo/ztensor/commit/879cbc9151670b523bc3138e7bdff2058c73137f))
* **graph:** add LMHead to nonCapturableOps ([07ba531](https://github.com/zerfoo/ztensor/commit/07ba531600509e290502b55c2c0ad603c382034f))
* **graph:** T4.1 add capture watchdog with 30s timeout and status sampling ([b3066a5](https://github.com/zerfoo/ztensor/commit/b3066a5f3d239f3ab1667df27adec9881e4c92f4))
* **graph:** T99.1.2 mark Gemma4PLECombinedProducer non-capturable ([6c855a9](https://github.com/zerfoo/ztensor/commit/6c855a928e8d8c96bd9de47728093a2c02547e41))


### Bug Fixes

* **graph:** T98.2.3 don't pool-release pass-through node inputs ([6ecf8db](https://github.com/zerfoo/ztensor/commit/6ecf8db036009e4839811df6f6a413fc297fda8f))

## [1.5.0](https://github.com/zerfoo/ztensor/compare/v1.4.0...v1.5.0) (2026-04-10)


### Features

* **compute:** add AllocDeviceFloat32 and CopyToDevice to FusedEncoderProvider ([8d6c90b](https://github.com/zerfoo/ztensor/commit/8d6c90bbc11342f9ccf021df25c6ee305c90e463))
* **compute:** add fused PatchTST encoder layer CUDA kernels ([4dfd46e](https://github.com/zerfoo/ztensor/commit/4dfd46e6a45b67c7de453cd30c25ec3716560112))


### Bug Fixes

* **compute:** GPUEngine.Reshape honors dst argument ([18a53fe](https://github.com/zerfoo/ztensor/commit/18a53fe45a63b388d52a86bdb32cbaf618f48a5c))
* **compute:** reuse dst GPU memory instead of allocating per call ([#84](https://github.com/zerfoo/ztensor/issues/84)) ([26bbd49](https://github.com/zerfoo/ztensor/commit/26bbd49de27be8811fa9e5e1176c5f1bbd6ea5eb))
* **kernels:** rename kernel_add in fused_encoder_bwd to avoid symbol clash ([716bbd6](https://github.com/zerfoo/ztensor/commit/716bbd6103eb9f96dc8eaeaba01b23d387ca4aed))

## [1.4.0](https://github.com/zerfoo/ztensor/compare/v1.3.0...v1.4.0) (2026-04-06)


### Features

* **graph:** add NewPJRTClient for external PJRT usage ([c8db036](https://github.com/zerfoo/ztensor/commit/c8db036b51820dae1a80886dbcca64f10328f945))
* **graph:** add PJRTPlan execution wrapper with KV cache state management ([3e5cb40](https://github.com/zerfoo/ztensor/commit/3e5cb40ca34b3f041c7eae25e30fbd16188f1482))


### Bug Fixes

* **ci:** exclude metal and pjrt from go vet ([5a7fdc3](https://github.com/zerfoo/ztensor/commit/5a7fdc30334582019f62721780fda29d62ab76bc))
* **kernels:** update GemvQ5_0F32 test to match qhOffset/qsOffset signature ([70f8fd5](https://github.com/zerfoo/ztensor/commit/70f8fd590785a0b76489c58619f431bb39f0c5b2))

## [1.3.0](https://github.com/zerfoo/ztensor/compare/v1.2.0...v1.3.0) (2026-04-03)


### Features

* **graph:** add CompilePJRT for PJRT backend compilation ([dfd77a4](https://github.com/zerfoo/ztensor/commit/dfd77a4f5631337f8ca8145d2b7f4bdcc3a1f807))
* **pjrt:** add buffer management (host-device transfer, readback, lifecycle) ([9b5dc75](https://github.com/zerfoo/ztensor/commit/9b5dc7552cafa8d5adf9845cba716490aeb4055a))
* **pjrt:** add KV cache I/O rewriting and executable cache ([c8decc5](https://github.com/zerfoo/ztensor/commit/c8decc52143a3f9796e7d82e65885ef525df6571))
* **pjrt:** add PJRT C API purego bindings for plugin loading, client, and device ([c675807](https://github.com/zerfoo/ztensor/commit/c675807eda36afc536e674f30eecb7ef81358006))
* **pjrt:** add program execution, serialization, and full StableHLO emitter ([382ea0a](https://github.com/zerfoo/ztensor/commit/382ea0ab9bc8e0adcdf15b63b5bcbe2313642238))
* **pjrt:** add StableHLO program compilation wrapper ([7fcdde7](https://github.com/zerfoo/ztensor/commit/7fcdde7d88bcbce353430965a98ee94cebef0b07))
* **stablehlo:** add emitter for element-wise and unary ops ([499cef2](https://github.com/zerfoo/ztensor/commit/499cef258811e227c714563c8185c31a8718f70f))
* **stablehlo:** add emitter for MatMul and structural ops ([13d87df](https://github.com/zerfoo/ztensor/commit/13d87df291ef872a5ab73f8334c441177068af4f))
* **stablehlo:** add emitter for reductions and Softmax decomposition ([c07b287](https://github.com/zerfoo/ztensor/commit/c07b287f657c1eb9ce5978e5bd94240fd7ccb7eb))
* **stablehlo:** add MLIR type system and SSA naming ([7c68d1e](https://github.com/zerfoo/ztensor/commit/7c68d1e19f3ba13c3d541f9ab6e85fe02b54ec32))
* **stablehlo:** add shape inference for arithmetic ops ([cac094e](https://github.com/zerfoo/ztensor/commit/cac094e507b9135e58f211094926fa68907e6458))
* **stablehlo:** add shape inference for structural ops ([8bf132c](https://github.com/zerfoo/ztensor/commit/8bf132cbf4443ff32e195e530cd1af1d0865399c))


### Bug Fixes

* **pjrt:** centralize internal/cuda import in pjrt.go ([aa8c170](https://github.com/zerfoo/ztensor/commit/aa8c1701a0f4f5fd5fc4c9c9621808075c59e8e8))
* **pjrt:** remove duplicate ccall/goStringN declarations ([3e5fba9](https://github.com/zerfoo/ztensor/commit/3e5fba995e92b565dcf23f132166553b27ee8c7e))

## [1.2.0](https://github.com/zerfoo/ztensor/compare/v1.1.3...v1.2.0) (2026-04-01)


### Features

* **cuda:** add Q6_K, Q5_K, Q5_0 GPU dequant kernels for M&gt;1 prefill ([d57e37e](https://github.com/zerfoo/ztensor/commit/d57e37edd9effe535125f30c45b3f03859bc57da))
* **cuda:** add Q8 Gather kernel for GPU embedding lookup ([30eb9c4](https://github.com/zerfoo/ztensor/commit/30eb9c4b79226b4461e75dde38819eb9735fef22))
* **tensor:** add QuantizeQ4K for float32 to Q4_K quantization ([d0d3a82](https://github.com/zerfoo/ztensor/commit/d0d3a82ea5bdeb5a92c0966275f59d9bdfd88c1b))


### Bug Fixes

* **compute:** add Q4KStorage to UploadWeights F32 skip list ([cc071b6](https://github.com/zerfoo/ztensor/commit/cc071b6ed4ae3398c0db2a9d8af1734db408a0b1))
* **compute:** CPU dequant fallback for Q4_K when K%256!=0 ([f50ffa7](https://github.com/zerfoo/ztensor/commit/f50ffa7b09eae8a88a2a594f04419d3af74d7720))
* **compute:** use dequant+cuBLAS for Q4_K when K%256!=0 ([5f21cbb](https://github.com/zerfoo/ztensor/commit/5f21cbbbb91d8e72677e637f0de09aff7452bd20))
* **compute:** use pool-backed GPUStorage for pool allocations ([4367330](https://github.com/zerfoo/ztensor/commit/43673306d83c4c1b089ca0932a26da81176ed695))
* **cuda:** byte-wise loads in Q5_0 GEMV for ARM64 alignment ([5f19e54](https://github.com/zerfoo/ztensor/commit/5f19e546feea3498de2b7b32b52f18fe9d2aa99f))
* **kernels:** check null function pointer in FusedSoftmaxVMulF32 ([935ad61](https://github.com/zerfoo/ztensor/commit/935ad613e27350a2c479ef071285aa0172f7b9ba))


### Performance Improvements

* **cuda:** separated GPU layout for Q5_0 GEMV ([d456c39](https://github.com/zerfoo/ztensor/commit/d456c3972f484634c3f0a8d6702d519aa7d748c1))

## [1.1.3](https://github.com/zerfoo/ztensor/compare/v1.1.2...v1.1.3) (2026-04-01)


### Bug Fixes

* **compute:** add Q5_0Storage B-weight handling to CPU MatMul ([e7927e5](https://github.com/zerfoo/ztensor/commit/e7927e57896d3b79d2d084eb0a71322ee608192c))
* **compute:** Q5_0 GEMV byte-wise loads for ARM64 alignment ([5c7ec7a](https://github.com/zerfoo/ztensor/commit/5c7ec7a7eeba9e7e691ca34e77adb03023e401b4))
* **compute:** skip Q4Storage in UploadWeights F32 loop (revert overaggressive skip) ([2e91650](https://github.com/zerfoo/ztensor/commit/2e91650f3aed1161c892bcd81cbb6dbe05e8f58f))
* **compute:** skip transpose reshape fast-path for square matrices ([eab19d0](https://github.com/zerfoo/ztensor/commit/eab19d078a49e38f88a4247ddbc96324b2c50595))

## [1.1.2](https://github.com/zerfoo/ztensor/compare/v1.1.1...v1.1.2) (2026-03-31)


### Bug Fixes

* **compute:** upload CPU fallback MatMul results to GPU for device consistency ([5bc914b](https://github.com/zerfoo/ztensor/commit/5bc914b9ece94d1c87ae12701d7c285f531c6273))

## [1.1.1](https://github.com/zerfoo/ztensor/compare/v1.1.0...v1.1.1) (2026-03-31)


### Bug Fixes

* **cuda:** remove float4 alignment requirement from gemv_q8_kernel ([1313605](https://github.com/zerfoo/ztensor/commit/13136054f73ebb31669f5509e06f79e3e8832009))
* **cuda:** remove float4 alignment requirement from gemv_q8_kernel ([34aba3b](https://github.com/zerfoo/ztensor/commit/34aba3b9c3b3458cf231a61dd2995b42dc2b394c))

## [1.1.0](https://github.com/zerfoo/ztensor/compare/v1.0.0...v1.1.0) (2026-03-31)


### Features

* **compute:** add GPUFusedSoftmaxVMul method with provider interface ([d659e76](https://github.com/zerfoo/ztensor/commit/d659e765ffb9a75ad95d7265a6e0b68fd590578e))
* **compute:** add GPURepeatInterleave method with purego bindings ([6af7b96](https://github.com/zerfoo/ztensor/commit/6af7b96b3dda59b665e7cb26d06374acb3ab941c))
* **compute:** add GraphCapturer interface for CUDA graph capture/replay ([1f37c69](https://github.com/zerfoo/ztensor/commit/1f37c699ccaef5ae8e9f5bee22d010262c6b0adb))
* **compute:** GPU-native Copy using cudaMemcpyAsync D2D ([efc8b42](https://github.com/zerfoo/ztensor/commit/efc8b42c4ba17e3d52152e0bbbff3deaa26d0ca4))
* **compute:** wire capture-aware pool into GPUEngine BeginCapture/EndCapture ([e39b318](https://github.com/zerfoo/ztensor/commit/e39b318506372c3b0b3da8a4981efe7584d764a2))
* **cuda:** add cudaMallocAsync and cudaFreeAsync bindings ([e339656](https://github.com/zerfoo/ztensor/commit/e339656ebe4c179cfcaaba3fe87d75c9984257f7))
* **cuda:** add cudaMemsetAsync binding and GPU-native Zero ([47b5d39](https://github.com/zerfoo/ztensor/commit/47b5d396dda24a2a9adf448e7b952f52a478150d))
* **cuda:** add fused repeat-interleave kernel for GQA head expansion ([91e2469](https://github.com/zerfoo/ztensor/commit/91e2469a6ac2b777a666fe58ca180f7e1db6edab))
* **cuda:** add fused softmax + V multiply kernel for decode attention ([ef6f7ce](https://github.com/zerfoo/ztensor/commit/ef6f7ce30104a8769870ff85ef31a6459d3648e8))
* **cuda:** make MemPool capture-aware with SetCaptureStream ([58b6337](https://github.com/zerfoo/ztensor/commit/58b63372c3100d560a3678d2f54493932e323065))
* **gpuapi:** wire FusedSoftmaxVMulF32 into KernelRunner interface ([9afdb01](https://github.com/zerfoo/ztensor/commit/9afdb01d9c2b0fcda08982e8df60034628684799))


### Bug Fixes

* **compute:** copy mmap bytes to heap in mmapDevicePtr fallback ([0ad23b5](https://github.com/zerfoo/ztensor/commit/0ad23b5572da224c0c56ef893e87eec005ba66ae))
* **compute:** revert H2D to sync Memcpy (async breaks mmap'd tensors) ([9a87e36](https://github.com/zerfoo/ztensor/commit/9a87e369f417a5ddfad21f23adbb8418412c6751))
* **compute:** use async memcpy in getDevicePtr for CUDA graph capture ([b36b7ed](https://github.com/zerfoo/ztensor/commit/b36b7ed88455bb9fb3b6649d4356679fcbf8d181))

## [1.0.0](https://github.com/zerfoo/ztensor/compare/v0.15.0...v1.0.0) (2026-03-30)


### Miscellaneous Chores

* release 1.0.0 ([0230a86](https://github.com/zerfoo/ztensor/commit/0230a86f12e84ea40459cd31681f68b9035455e7))

## [0.15.0](https://github.com/zerfoo/ztensor/compare/v0.14.1...v0.15.0) (2026-03-29)


### Features

* **tensor:** MmapStorage.SliceElements for zero-copy expert weight slicing ([0a40e11](https://github.com/zerfoo/ztensor/commit/0a40e11c698406358918dafd9401782ce0d43f71))
* **xblas:** streaming GEMM for mmap'd tensors, unblocks over-RAM inference ([8d80b91](https://github.com/zerfoo/ztensor/commit/8d80b914343916b3e1f9a578630bbf129515d405))

## [0.14.1](https://github.com/zerfoo/ztensor/compare/v0.14.0...v0.14.1) (2026-03-28)


### Bug Fixes

* **ci:** exclude purego GPU binding packages from go vet ([60f0f66](https://github.com/zerfoo/ztensor/commit/60f0f6602adc0ca602cb2cc026835f004c8b4a36))
* **tensor:** add IQ3_S to quant registry expected list ([98c9237](https://github.com/zerfoo/ztensor/commit/98c9237886ff8bdb8f44fd0786e6a311aa121cd9))

## [0.14.0](https://github.com/zerfoo/ztensor/compare/v0.13.0...v0.14.0) (2026-03-28)


### Features

* **graph:** add NodeOutput method for intermediate activation extraction ([76a29c6](https://github.com/zerfoo/ztensor/commit/76a29c62546c68faa35bbcfb74d694288c8f84b3))

## [0.13.0](https://github.com/zerfoo/ztensor/compare/v0.12.0...v0.13.0) (2026-03-28)


### Features

* **xblas:** add fused Q4_K GEMV kernel — 17x faster than dequant+requant ([7ceb267](https://github.com/zerfoo/ztensor/commit/7ceb26795f37fa9fcc741847cb991ded97c208bc))

## [0.12.0](https://github.com/zerfoo/ztensor/compare/v0.11.0...v0.12.0) (2026-03-28)


### Features

* **tensor:** make TernaryStorage implement Storage[float32] ([2c8e9fa](https://github.com/zerfoo/ztensor/commit/2c8e9fa8fe0b9f2e06f338be611a77a9ba290ffa))


### Bug Fixes

* **kernels:** add missing NSA, KV dequant, and IQ dequant fields to KernelLib ([bf32aef](https://github.com/zerfoo/ztensor/commit/bf32aef6c977330750c8afb3f5d7c032e9e43c21))

## [0.11.0](https://github.com/zerfoo/ztensor/compare/v0.10.1...v0.11.0) (2026-03-27)


### Features

* **compute:** add CosineSimilarity to Engine[T] ([204f07b](https://github.com/zerfoo/ztensor/commit/204f07b49fc8d981dfb8d80539beec8548889e9a))
* **compute:** add GPU dispatch for CosineSimilarity ([40588bc](https://github.com/zerfoo/ztensor/commit/40588bcbb249a35e5b8973b5aea813cb14810cc5))
* **compute:** add GPU dispatch for ternary GEMV ([295f61c](https://github.com/zerfoo/ztensor/commit/295f61ceddab8ec5ee367dd51c82c9059792c7d7))
* **compute:** add Hadamard matrix generator ([b3b3478](https://github.com/zerfoo/ztensor/commit/b3b347838f6b2c21a4c61761bc194a98180a20ad))
* **compute:** add HadamardTransform to Engine[T] ([5a99614](https://github.com/zerfoo/ztensor/commit/5a99614839b20deca6fc37095209c7ee1e086f5c))
* **compute:** add ReduceMax to Engine[T] ([4b9b712](https://github.com/zerfoo/ztensor/commit/4b9b712e47381fbeabbf68d2cc42dfe2237e0531))
* **compute:** add split-KV flash decode kernel with CPU reference ([c16817e](https://github.com/zerfoo/ztensor/commit/c16817e6211fa8d6e6b19bf751f71e6918d434f5))
* **compute:** add split-KV flash decode kernel with CPU reference ([41feddf](https://github.com/zerfoo/ztensor/commit/41feddfbccfbe7e79daa59b5d3a671daff31e919))
* **compute:** add TernaryGEMV for ternary weight matrix-vector multiply ([8731bd1](https://github.com/zerfoo/ztensor/commit/8731bd172c60590fb0df32c93c4ffa4fdab50d9b))
* **cuda:** add fused NSA three-path attention kernel stub ([a024958](https://github.com/zerfoo/ztensor/commit/a024958cf5f5acb3398c5724476068a6e37b0bc5))
* **tensor:** add IQ2_XXS dequantization storage ([48677a7](https://github.com/zerfoo/ztensor/commit/48677a73b9937ec17806e3df8de8a4b335a5d350))
* **tensor:** add IQ3_S dequantization storage ([9eab58b](https://github.com/zerfoo/ztensor/commit/9eab58b77dd7a5fe70e2027485e3a7f440670de0))
* **tensor:** add IQ4_NL dequantization storage ([5205837](https://github.com/zerfoo/ztensor/commit/5205837c8fd596f7f621bd27e7ac906865b1346e))
* **tensor:** add TernaryStorage for 2-bit ternary weights ([0f7c5ca](https://github.com/zerfoo/ztensor/commit/0f7c5ca4350fcb3baa4b1f6ecaf6c33b3de8fc5b))

## [0.10.1](https://github.com/zerfoo/ztensor/compare/v0.10.0...v0.10.1) (2026-03-27)


### Bug Fixes

* **tensor:** remove MADV_SEQUENTIAL from MmapFile (caused 7x load regression) ([8949a19](https://github.com/zerfoo/ztensor/commit/8949a19d131f32efddcc68bf1571cb44d91ae93a))

## [0.10.0](https://github.com/zerfoo/ztensor/compare/v0.9.6...v0.10.0) (2026-03-27)


### Features

* **tensor:** add madvise hints for mmap'd pages ([e26c8d6](https://github.com/zerfoo/ztensor/commit/e26c8d6a56e5da3959d98b3c2158e1e353e9af26))

## [0.9.6](https://github.com/zerfoo/ztensor/compare/v0.9.5...v0.9.6) (2026-03-27)


### Bug Fixes

* **graph:** skip all quantized storage in EnsureSlotsGPU/EnsureCaptureInputsGPU ([0b38668](https://github.com/zerfoo/ztensor/commit/0b38668bcfb6125270f56e95d3421ce36345bf37))

## [0.9.5](https://github.com/zerfoo/ztensor/compare/v0.9.4...v0.9.5) (2026-03-27)


### Bug Fixes

* **compute:** skip MmapStorage entirely in UploadWeights ([8796fd0](https://github.com/zerfoo/ztensor/commit/8796fd03f6ee406c92900eab6cef0662d338ba1e))

## [0.9.4](https://github.com/zerfoo/ztensor/compare/v0.9.3...v0.9.4) (2026-03-27)


### Bug Fixes

* **compute:** copy mmap bytes to heap before cudaMemcpy upload ([c2d68e7](https://github.com/zerfoo/ztensor/commit/c2d68e7e860d92410b36cfb26a2772b2753bfa5c))

## [0.9.3](https://github.com/zerfoo/ztensor/compare/v0.9.2...v0.9.3) (2026-03-27)


### Bug Fixes

* **graph:** skip quantized storage in PreUploadFrozenWeights ([4b8388c](https://github.com/zerfoo/ztensor/commit/4b8388c9ad14d2af78d2e2695a33df12028dd19b))

## [0.9.2](https://github.com/zerfoo/ztensor/compare/v0.9.1...v0.9.2) (2026-03-27)


### Bug Fixes

* **compute:** skip F32 MmapStorage in quantized upload path ([51ed3e7](https://github.com/zerfoo/ztensor/commit/51ed3e7090f0a44215a9b3ddc56b460128c0fc08))

## [0.9.1](https://github.com/zerfoo/ztensor/compare/v0.9.0...v0.9.1) (2026-03-27)


### Bug Fixes

* **tensor:** delegate K-quant MmapStorage dequant to reference implementations ([3ef8261](https://github.com/zerfoo/ztensor/commit/3ef8261f40c8cc58ce01bf5eca31948f143dbf2e))

## [0.9.0](https://github.com/zerfoo/ztensor/compare/v0.8.0...v0.9.0) (2026-03-27)


### Features

* **compute:** add MmapStorage GPU dispatch for quantized GEMV/GEMM ([62f3db1](https://github.com/zerfoo/ztensor/commit/62f3db145b46446bffd0f22ad041e639e1c1a483))

## [0.8.0](https://github.com/zerfoo/ztensor/compare/v0.7.0...v0.8.0) (2026-03-27)


### Features

* **tensor:** add Q4_1/Q5_0/Q5_1 support for MmapStorage ([8adb879](https://github.com/zerfoo/ztensor/commit/8adb87914d5d48ae2c54db6368ac2a83a1ce88d8))

## [0.7.0](https://github.com/zerfoo/ztensor/compare/v0.6.3...v0.7.0) (2026-03-27)


### Features

* **tensor:** add MmapStorage type and platform mmap helpers ([f8b48bb](https://github.com/zerfoo/ztensor/commit/f8b48bb901fe9dd46a25a40b1eff05e92a7ab9bd))

## [0.6.3](https://github.com/zerfoo/ztensor/compare/v0.6.2...v0.6.3) (2026-03-27)


### Bug Fixes

* **compute:** change Repeat to repeat-each semantics for GQA correctness ([d3e6b96](https://github.com/zerfoo/ztensor/commit/d3e6b9667ad578080ce5f3931d84848b21c79eb5))

## [0.6.2](https://github.com/zerfoo/ztensor/compare/v0.6.1...v0.6.2) (2026-03-26)


### Bug Fixes

* **compute:** prevent FP16 MatMul segfault on aarch64 purego ([a6756c5](https://github.com/zerfoo/ztensor/commit/a6756c5446009370a0d7ca4629f72b3403fe1df9))

## [0.6.1](https://github.com/zerfoo/ztensor/compare/v0.6.0...v0.6.1) (2026-03-26)


### Bug Fixes

* **compute:** add VRAM bounds check for large MatMul allocations ([915816c](https://github.com/zerfoo/ztensor/commit/915816c739febfccf580ec6d47df2941b33677f5))

## [0.6.0](https://github.com/zerfoo/ztensor/compare/v0.5.1...v0.6.0) (2026-03-26)


### Features

* **gguf:** add shared GGUF writer package ([0709c09](https://github.com/zerfoo/ztensor/commit/0709c096728e59a41a9760cbf91cd10f66ef147f))

## [0.5.1](https://github.com/zerfoo/ztensor/compare/v0.5.0...v0.5.1) (2026-03-26)


### Bug Fixes

* **cuda:** raise shared memory limit for Q4 GEMV with K &gt; 12288 ([d654c72](https://github.com/zerfoo/ztensor/commit/d654c725b6919d8aa63f85266f009cf326d6375d))

## [0.5.0](https://github.com/zerfoo/ztensor/compare/v0.4.1...v0.5.0) (2026-03-25)


### Features

* **tensor:** add MergeQ4KStorage and MergeQ6KStorage ([764a750](https://github.com/zerfoo/ztensor/commit/764a750b10af622d5655ad7342a52cd8d574e5ef))

## [0.4.1](https://github.com/zerfoo/ztensor/compare/v0.4.0...v0.4.1) (2026-03-24)


### Bug Fixes

* **cuda:** expand libkernels.so search paths and log dlopen errors (issue [#7](https://github.com/zerfoo/ztensor/issues/7)) ([b906f08](https://github.com/zerfoo/ztensor/commit/b906f08ac3424a4ff4ab34fe888014a009ee2122))

## [0.4.0](https://github.com/zerfoo/ztensor/compare/v0.3.2...v0.4.0) (2026-03-24)


### Features

* add Q5_0 fused dequant-GEMV kernel stack ([de5331f](https://github.com/zerfoo/ztensor/commit/de5331f46d394e49078ead9f279d98690b2ff4da))
* add Q5_K fused dequant-GEMV kernel stack ([c2ea6f7](https://github.com/zerfoo/ztensor/commit/c2ea6f76f9125c85e59f5d2f872d8b5b3a539ebb))
* **batched:** add batched multi-model inference ([5897e29](https://github.com/zerfoo/ztensor/commit/5897e29cbda31d83015bf9a84c43d032bbb504cf))
* **compute:** add ComputeAmax and ScaleForFP8 for FP8 quantization (T2.2) ([8c866f4](https://github.com/zerfoo/ztensor/commit/8c866f412818f2d2becf873e056543b0da257cd5))
* **compute:** add native Q5_K GEMV kernel ([b428f17](https://github.com/zerfoo/ztensor/commit/b428f1706da62b09b090b7433838d012303e8a4b))
* **compute:** add Q6_K GEMV dispatch and GPU engine integration ([0528588](https://github.com/zerfoo/ztensor/commit/052858826b42f9798d692a9a44b9e8b5fee3ef60))
* **compute:** dispatch FP8 MatMul to cublasLt FP8 GEMM (T2.3) ([c446655](https://github.com/zerfoo/ztensor/commit/c4466559a5859eb9763a64366c8527050b6ba1ca))
* **compute:** FP16 weight upload path + PreUploadFrozenWeights skip ([d893b9c](https://github.com/zerfoo/ztensor/commit/d893b9c3ceb38138f36680524d4862305dae52d4))
* **compute:** implement hardware profiling and detection framework ([c0c7ef5](https://github.com/zerfoo/ztensor/commit/c0c7ef55f9d926e42cca3e91ef5180d5bc3bdd23))
* **compute:** wire paged attention into GQA (T1.4) ([abeff7a](https://github.com/zerfoo/ztensor/commit/abeff7a89fa11e2c3d490708f07ca7c38668eaa9))
* **cuda:** add FP8 GEMM kernel with cublasLt bindings (T2.1) ([7f524bc](https://github.com/zerfoo/ztensor/commit/7f524bc59a0ed6e56e3d78ef014599cef15c3590))
* **cuda:** add NVFP4 GEMV kernel for Blackwell sm_100+ (T2.5) ([63fad59](https://github.com/zerfoo/ztensor/commit/63fad5991644ee0c73a94aa01468a5b1051aca6d))
* **cuda:** add paged attention kernel with block-table indirection (T1.3) ([e89e01d](https://github.com/zerfoo/ztensor/commit/e89e01de0ef548a17e7cf0fd4b73a04c386dbd02))
* **cuda:** add Q6_K fused dequant-GEMV kernel ([8fc89db](https://github.com/zerfoo/ztensor/commit/8fc89dbf751d1e3d024529e2f4077823df2a3151))
* **cuda:** add ragged batching attention kernel (T1.6) ([2748ebc](https://github.com/zerfoo/ztensor/commit/2748ebcadbffda0ab493d8775791d427ea792398))
* **cuda:** add selective scan kernel for Mamba/SSM (T6.1) ([260160e](https://github.com/zerfoo/ztensor/commit/260160ea661e4c67a30b8e72b8ad0c327913424e))
* **cuda:** implement FlashAttention-2 fused kernel with GQA support ([e7000f8](https://github.com/zerfoo/ztensor/commit/e7000f8ce110d99c02f765b54dabe6fd979b4dcc))
* **cuda:** implement warp-specialized GEMV kernel for decode phase ([fc46cab](https://github.com/zerfoo/ztensor/commit/fc46cab969f511fd96726890b97533c59adb3a53))
* **cuda:** optimize Q4_K GEMV for sm_121 (Blackwell GB10 / DGX Spark) ([3e32432](https://github.com/zerfoo/ztensor/commit/3e324328154a2a8a708309f296228fbeaac48e42))
* **fpga:** add FPGA runtime abstraction layer via purego ([e703a86](https://github.com/zerfoo/ztensor/commit/e703a8675fbec560bd5e4d42fc9362eb2bb8bc09))
* **gpuapi:** implement Apple Metal compute shader bindings via purego ([d548e22](https://github.com/zerfoo/ztensor/commit/d548e22f792371b21ba1b6e4515334adbb9b27a2))
* **gpuapi:** implement Apple Metal compute shader bindings via purego ([38212db](https://github.com/zerfoo/ztensor/commit/38212db1c9913d7edcbbc244f41900300f2213fe))
* **graph:** add fast replay path skipping PrepareSlots/EnsureSlotsGPU ([e6e2355](https://github.com/zerfoo/ztensor/commit/e6e23557922ef51fdff07b25bfe3b53ec6889866))
* **graph:** add gradient checkpointing (T8.9) ([3cd5c01](https://github.com/zerfoo/ztensor/commit/3cd5c019aefad037b9168641acb055c0c7ca164d))
* **graph:** add kernel launch batch scheduler ([cfd513b](https://github.com/zerfoo/ztensor/commit/cfd513bcb535a2998a738211f4430fc636b54c46))
* **graph:** add SaveParameters/LoadParametersFromFile and checkpoint serialization ([8a930ec](https://github.com/zerfoo/ztensor/commit/8a930ecce3365223cad5904b51a7a17cdcec6a8c)), closes [#96](https://github.com/zerfoo/ztensor/issues/96)
* **graph:** add SlotCount method to ExecutionPlan ([b8dc85f](https://github.com/zerfoo/ztensor/commit/b8dc85f943112dc713f148e6c474b3168a05f983))
* **graph:** cache EmbeddingLookup GPU buffer for fast replay ([dc595dd](https://github.com/zerfoo/ztensor/commit/dc595dd879e1cc89e0474674e8dd54b4bd771518))
* **graph:** expand CUDA graph capture to 100% instruction coverage ([33b54d9](https://github.com/zerfoo/ztensor/commit/33b54d9ea379cd37b7f0e04ecf4b78910edf49a8))
* **kv:** add BlockPool for paged attention (T1.1) ([e851d47](https://github.com/zerfoo/ztensor/commit/e851d47ca1042278709dbf7b7d675c7957ff5c87))
* **kv:** add BlockTable for per-sequence paged KV mapping (T1.2) ([be1ff30](https://github.com/zerfoo/ztensor/commit/be1ff30d6c25a1d10dbd76b82e4ed50398c59aea))
* **kv:** add RadixTree for KV block prefix caching (T4.1) ([0e68dc9](https://github.com/zerfoo/ztensor/commit/0e68dc9f3d9b5e4ea33f764d46055401ca868789))
* **metal:** port critical CUDA kernels to Metal compute shaders ([3051613](https://github.com/zerfoo/ztensor/commit/3051613288e9472a3a83c691c9e69ca34ca11104))
* **metrics:** add Add(n int64) to CounterMetric interface ([64728d8](https://github.com/zerfoo/ztensor/commit/64728d8275133c4aea5d5d4fc7aa7ca1c0e31ffc))
* **quant:** add native Q6_K GEMV direct decode for CPU and CUDA ([566136b](https://github.com/zerfoo/ztensor/commit/566136bb494890ad3761a5e4306681bdc81ac612))
* **quant:** add W4A16 mixed-precision dispatch ([8d2f97a](https://github.com/zerfoo/ztensor/commit/8d2f97a50579c42664c5799569a43404c882771c))
* **quant:** add W8A8 mixed-precision dispatch with INT8 weights/activations and FP32 accumulation ([3fe0745](https://github.com/zerfoo/ztensor/commit/3fe0745d277445c119a2becf55283130bea2a52e))
* **sycl:** add SYCL runtime bindings via purego ([b987c36](https://github.com/zerfoo/ztensor/commit/b987c369bd23c6cdedd0747f2c99e8b5bb2a527d))
* **sycl:** port GEMV and attention kernels to SYCL backend ([61b0ee8](https://github.com/zerfoo/ztensor/commit/61b0ee81cebca056e1a0b0d3d2a53b70b02d6e77))
* **tensor:** add AWQ dequantization support ([cfbc3d0](https://github.com/zerfoo/ztensor/commit/cfbc3d04d2cf0e920c7848ba8bf174eb25f8d5c3))
* **tensor:** add NewFloat16StorageFromRaw constructor for pre-encoded FP16 bytes ([d21c355](https://github.com/zerfoo/ztensor/commit/d21c3558e37b43461579dab451ef7c6caa00fa81))
* **tensor:** add NewFloat16StorageFromRaw for FP16 GGUF loader ([fbb968d](https://github.com/zerfoo/ztensor/commit/fbb968d00ea26023abbdb5f171a4e3c41bcf41ac))
* **tensor:** add NF4 quantization with double quantization (T9.3) ([beaba05](https://github.com/zerfoo/ztensor/commit/beaba05610b45f88a2a5421706486e116bf9e0ad))
* **tensor:** add NVFP4 E2M1 weight storage (T2.4) ([6f630dd](https://github.com/zerfoo/ztensor/commit/6f630dda40020c261d36679f9d37963d2584bafe))
* **tensor:** add NVFP4 E2M1 weight storage (T2.4) ([ccd48ec](https://github.com/zerfoo/ztensor/commit/ccd48ecc19206001ef847331e8015ee798b9ad49))
* **tensor:** implement GPTQ dequantization ([3784403](https://github.com/zerfoo/ztensor/commit/37844039eb9cffc2b0dc0a5efc613283c87d0bba))
* **tensor:** implement quantization format registry ([f501c21](https://github.com/zerfoo/ztensor/commit/f501c21fa6cbd2572fb94f1829a30d145be061c8))
* **tensorrt:** add TensorRT compilation for tabular models ([90f408a](https://github.com/zerfoo/ztensor/commit/90f408a205114ed281b16340ee1e3a9094d9d97a))


### Bug Fixes

* **cuda:** add gemv_q4k_sm121.cu to kernel build sources (issue [#7](https://github.com/zerfoo/ztensor/issues/7)) ([0324568](https://github.com/zerfoo/ztensor/commit/0324568a856fcf47e2007de4202732e8f6199282))
* **cuda:** dispatch Q4_K GEMV directly on sm_121 without re-quantization ([10349fe](https://github.com/zerfoo/ztensor/commit/10349fe033204ea0a44d8dfffae41aa54bd304d6))
* **cuda:** replace cgo_import_dynamic JMP trampolines with runtime.dlopen on arm64 ([38f54ab](https://github.com/zerfoo/ztensor/commit/38f54ab42b6d762452758df39267409e61a72b21)), closes [#3](https://github.com/zerfoo/ztensor/issues/3)
* **cuda:** resolve Q5_K_M and Q6_K quantized GEMM/GEMV test failures ([488862c](https://github.com/zerfoo/ztensor/commit/488862cc4bc955f7dfe76dee459356bd4605feb7))
* **cuda:** use cgo build tag for arm64 dlopen trampolines ([ebff59e](https://github.com/zerfoo/ztensor/commit/ebff59e74a8d9429c6715a342be995346ed79305))
* **gemv:** remove unused dp4a accumulator variables ([3653fe1](https://github.com/zerfoo/ztensor/commit/3653fe18cd9c903a9ea80ee61d803c13d8e0cee2))
* **graph:** remove Q4Storage skip — restore cuBLAS SGEMM path (188 tok/s) ([a38af9a](https://github.com/zerfoo/ztensor/commit/a38af9a202973cc550cf56862115359f61ea7fb2))
* **graph:** restore PreUploadFrozenWeights for stable 188 tok/s baseline ([2decc08](https://github.com/zerfoo/ztensor/commit/2decc0864b30c484aa5f9ac6473e0315747ffd6c))
* **graph:** skip BFloat16Storage in PreUploadFrozenWeights ([7da3407](https://github.com/zerfoo/ztensor/commit/7da340779983b822990c826a2e42e23caa2061af))
* **graph:** skip CUDA graph capture during prefill (seqLen &gt; 1) ([e5f9ce0](https://github.com/zerfoo/ztensor/commit/e5f9ce00af6cb3368911319dcb169fda4f40c8cd))
* **graph:** skip K-quant storage types in PreUploadFrozenWeights ([23ba86d](https://github.com/zerfoo/ztensor/commit/23ba86d1d7629ba82e7816b0e56412296f77008e))
* **graph:** skip Q4Storage in EnsureCaptureInputsGPU ([e4d4613](https://github.com/zerfoo/ztensor/commit/e4d4613b10def3ded1bbed4897b641d2e8c36173))
* **graph:** skip quantized tensors with GPU pointers in PreUploadFrozenWeights ([a7e361c](https://github.com/zerfoo/ztensor/commit/a7e361cefdbd5effdae67009c3c778a1c20fc6e5))
* **graph:** sort Parameters() by name and add LoadParameters method ([c1b853b](https://github.com/zerfoo/ztensor/commit/c1b853b21863252288d6582a34f4dc13512ef037))
* **tensor:** add missing NF4Storage implementation (T9.3 agent omitted impl) ([1e4beaa](https://github.com/zerfoo/ztensor/commit/1e4beaaa7c8b5154a2c725c95910fc3254e12ca7))


### Performance Improvements

* **arena:** add free-list for intra-pass buffer reuse ([d40d6e4](https://github.com/zerfoo/ztensor/commit/d40d6e4fc4025dff0bfa75e559d7cdc1e540a06f))
* clean Q4 GEMV restore — skip Q4 in PreUploadFrozenWeights + UploadWeights ([e6a6e30](https://github.com/zerfoo/ztensor/commit/e6a6e30f228862104d4312e2edcccafa5816d3e9))
* **compute:** convert Q4Storage to BF16 during upload (targeted, not all tensors) ([39c77c9](https://github.com/zerfoo/ztensor/commit/39c77c94dd1647167cb711af21910963fb5dc37f))
* **compute:** convert Q8 to float32 in UploadWeights for cuBLAS path ([4d4bd8d](https://github.com/zerfoo/ztensor/commit/4d4bd8d646b5ac4d2f7f3c8afff6ff134d387ec0))
* **compute:** upload large weight tensors as BF16 instead of F32 ([e43f03f](https://github.com/zerfoo/ztensor/commit/e43f03f886b535b3dc55f00107472068813b0726))
* **gemv:** add dp4a INT8 Q4_K GEMV kernel ([05c3113](https://github.com/zerfoo/ztensor/commit/05c3113fdd65a4e333b00bcb8a6c5f95ce237efa))
* **gemv:** prefer dp4a Q4_K GEMV when available ([ea98b7c](https://github.com/zerfoo/ztensor/commit/ea98b7c01ff8e34df78791393827eae42a18078e))
* **gemv:** reduce dp4a kernel register pressure ([cc707d5](https://github.com/zerfoo/ztensor/commit/cc707d585e4af38294b66992272931fd6ae0d54c))
* **gemv:** wire dp4a Q4_K GEMV kernel into purego loader ([8be7d1f](https://github.com/zerfoo/ztensor/commit/8be7d1f6d1be3b9b296b2e195f17979ad911d98e))
* **graph:** add tensor lifetime analysis and intra-pass arena reuse ([18e5f37](https://github.com/zerfoo/ztensor/commit/18e5f370bc61fae33c5b85a86e8c5ac1e29e078b))
* **graph:** let PreUploadFrozenWeights dequantize all quant types to float32 ([fd755b4](https://github.com/zerfoo/ztensor/commit/fd755b4982ce70d07462f7224053fd8d454c3ab2))
* **graph:** remove PreUploadFrozenWeights from CUDA graph executor ([adb6e1c](https://github.com/zerfoo/ztensor/commit/adb6e1cc0fb397d483c849519a4a4122a337f5b0))
* **graph:** skip Q4Storage in PreUploadFrozenWeights for Q4 GEMV path ([880b50e](https://github.com/zerfoo/ztensor/commit/880b50ec4feedb4671eab659439065881f839f71))
* restore Phase 6-compatible upload paths ([2cc6cc3](https://github.com/zerfoo/ztensor/commit/2cc6cc3e83304afb0f2839a8e3beb5f11b123130))
* restore Q4 GEMV path — skip Q4→F32 in both UploadWeights and PreUploadFrozenWeights ([f6faf2a](https://github.com/zerfoo/ztensor/commit/f6faf2a3713a91d6f0b9e83f2bf7e5e82bf7933c))
* **transpose:** restore Phase 6 GPU transpose guard ([aa0541b](https://github.com/zerfoo/ztensor/commit/aa0541bb77376d8031376f5b58ab7029d71fdf62))
* **ztensor:** dp4a Q4K GEMV kernel + arena free-list intra-pass reuse ([4e85b12](https://github.com/zerfoo/ztensor/commit/4e85b12f517a0e6fc04c75d9d656ca37463de613))


### Reverts

* **graph:** remove gpuPtrHolder check from PreUploadFrozenWeights ([dafb96e](https://github.com/zerfoo/ztensor/commit/dafb96e46fbda238e9811ea9b022fee4b3eac8bb))

## [0.3.2](https://github.com/zerfoo/ztensor/compare/v0.3.1...v0.3.2) (2026-03-21)


### Bug Fixes

* **cuda:** use cgo build tag for arm64 dlopen trampolines ([ebff59e](https://github.com/zerfoo/ztensor/commit/ebff59e74a8d9429c6715a342be995346ed79305))

## [0.3.1](https://github.com/zerfoo/ztensor/compare/v0.3.0...v0.3.1) (2026-03-21)


### Bug Fixes

* **cuda:** replace cgo_import_dynamic JMP trampolines with runtime.dlopen on arm64 ([38f54ab](https://github.com/zerfoo/ztensor/commit/38f54ab42b6d762452758df39267409e61a72b21)), closes [#3](https://github.com/zerfoo/ztensor/issues/3)
