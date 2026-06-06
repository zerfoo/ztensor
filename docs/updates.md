# ztensor session updates

## 2026-06-06 -- #106 REOPENED: chunking is not the fix

The v1.8.1 chunked bulkUploadF32 (64 MiB + 4096-tensor cap) is a defensive
bound but does NOT resolve #106. Wolf train-crossasset rebuilt against the
merged code still wedged the GB10 driver identically at the 213,304-tensor
scale (uninterruptible D-state). The wedge does not correlate with single-alloc
size; the exact CUDA ioctl is still unpinned.

Correction: the earlier "validated on GB10 / unblocked" claim came from a
256-tensor test that never reproduced the 213k-scale wedge.

Next: out-of-band watchdog to pin the wedging ioctl (plan E4). Blocked on
go-ahead -- deliberately wedging the shared GB10 risks an unkillable pod / host
restart.
