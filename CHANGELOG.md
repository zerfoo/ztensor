# Changelog

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
