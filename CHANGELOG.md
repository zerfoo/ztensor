# Changelog

## [0.3.2](https://github.com/zerfoo/ztensor/compare/v0.3.1...v0.3.2) (2026-03-21)


### Bug Fixes

* **cuda:** use cgo build tag for arm64 dlopen trampolines ([ebff59e](https://github.com/zerfoo/ztensor/commit/ebff59e74a8d9429c6715a342be995346ed79305))

## [0.3.1](https://github.com/zerfoo/ztensor/compare/v0.3.0...v0.3.1) (2026-03-21)


### Bug Fixes

* **cuda:** replace cgo_import_dynamic JMP trampolines with runtime.dlopen on arm64 ([38f54ab](https://github.com/zerfoo/ztensor/commit/38f54ab42b6d762452758df39267409e61a72b21)), closes [#3](https://github.com/zerfoo/ztensor/issues/3)
