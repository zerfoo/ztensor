#!/usr/bin/env python3
"""PyTorch-as-oracle runner for ztensor case bundles (zerfoo ADR 091, T1.3).

Reads the case bundles produced by the Go writer
(go run github.com/zerfoo/ztensor/testing/oracle/cmd/oracle-gen), replays each
op in PyTorch, backprops the recorded upstream gradient, and diffs torch's
forward output and gradients against the recorded ztensor ones within the
per-op tolerances from each manifest.

Designed to run OFFLINE inside nvcr.io/nvidia/pytorch:26.02-py3 on the DGX
GB10 via a Spark pod (see oracle-pod.yaml and README.md in this directory).
Dependencies: Python stdlib + numpy + torch only.

Bundle format (format_version 1): one directory per op containing
manifest.json plus raw little-endian row-major tensor files; see the Go
package testing/oracle (bundle.go) for the authoritative spec. The pass
criterion per element is |got - ref| <= atol + rtol * |ref| (ref = torch);
any NaN fails. This logic is mirrored by testing/oracle/diff.go, whose CI
red-proof test (fast-math tanh fixture) proves the report semantics.

Exit status: 0 if every bundle passed, 1 if any failed or errored.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

FORMAT_VERSION = 1

DTYPES = {
    "float32": (np.dtype("<f4"), torch.float32),
    "float64": (np.dtype("<f8"), torch.float64),
}


def load_tensor(bundle_dir, ref, device):
    """Reconstruct one tensor file as a torch tensor on device."""
    np_dtype, torch_dtype = DTYPES[ref["dtype"]]
    path = os.path.join(bundle_dir, ref["file"])
    arr = np.fromfile(path, dtype=np_dtype)
    expected = int(np.prod(ref["shape"])) if ref["shape"] else 1
    if arr.size != expected:
        raise ValueError(
            f"{ref['file']}: {arr.size} elements, shape {ref['shape']} wants {expected}"
        )
    arr = arr.reshape(ref["shape"])
    return torch.from_numpy(arr.copy()).to(device=device, dtype=torch_dtype)


def diff_stats(got, ref, atol, rtol):
    """Elementwise |got-ref| <= atol + rtol*|ref|; NaN anywhere fails.

    Mirrors testing/oracle/diff.go exactly. `got` is the recorded ztensor
    tensor, `ref` is torch's output (the oracle).
    """
    got = np.asarray(got, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    if got.size != ref.size:
        raise ValueError(f"size mismatch: got {got.size}, ref {ref.size}")
    nan_mask = np.isnan(got) | np.isnan(ref)
    abs_diff = np.abs(got - ref)
    rel_diff = abs_diff / np.maximum(np.abs(ref), 1e-12)
    fail_mask = nan_mask | (abs_diff > atol + rtol * np.abs(ref))
    mismatches = int(np.count_nonzero(fail_mask))
    return {
        "max_abs": math.inf if nan_mask.any() else float(abs_diff.max(initial=0.0)),
        "max_rel": math.inf if nan_mask.any() else float(rel_diff.max(initial=0.0)),
        "checked": int(got.size),
        "mismatches": mismatches,
        "pass": mismatches == 0,
    }


def run_bundle(bundle_dir, device):
    """Replay one bundle in torch and diff against the recorded ztensor data."""
    with open(os.path.join(bundle_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"format_version {manifest.get('format_version')} unsupported "
            f"(runner supports {FORMAT_VERSION})"
        )

    tol = manifest["tolerance"]
    namespace = {"torch": torch}
    leaves = []  # (kind, name, grad-ref) aligned with manifest order

    for ref in manifest["inputs"]:
        t = load_tensor(bundle_dir, ref, device).requires_grad_(True)
        namespace[ref["name"]] = t
        leaves.append(("input", ref["name"], t))
    for ref in manifest.get("params") or []:
        t = load_tensor(bundle_dir, ref, device).requires_grad_(True)
        namespace[ref["name"]] = t
        leaves.append(("param", ref["name"], t))

    upstream = load_tensor(bundle_dir, manifest["upstream"], device)

    # The expression is data, not user input: it comes from the checked-in Go
    # mapping table (testing/oracle/torchmap.go).
    y = eval(manifest["torch_expr"], {"__builtins__": {}}, namespace)  # noqa: S307
    y.backward(gradient=upstream.reshape(y.shape))

    result = {"op": manifest["op"], "torch_expr": manifest["torch_expr"], "grads": []}

    recorded_fwd = load_tensor(bundle_dir, manifest["forward"], device="cpu")
    result["forward"] = diff_stats(
        recorded_fwd.numpy(), y.detach().cpu().numpy(), tol["fwd_atol"], tol["fwd_rtol"]
    )

    grad_refs = {("input", ref["name"]): ref for ref in manifest["input_grads"]}
    for ref in manifest.get("params") or []:
        grad_refs[("param", ref["name"])] = dict(ref, file=ref["grad_file"])

    ok = result["forward"]["pass"]
    for kind, name, leaf in leaves:
        ref = grad_refs[(kind, name)]
        if leaf.grad is None:
            raise ValueError(f"torch produced no gradient for {kind} {name}")
        recorded = load_tensor(bundle_dir, ref, device="cpu")
        stats = diff_stats(
            recorded.numpy(),
            leaf.grad.detach().cpu().numpy(),
            tol["grad_atol"],
            tol["grad_rtol"],
        )
        stats["name"] = name
        stats["kind"] = kind
        result["grads"].append(stats)
        ok = ok and stats["pass"]

    result["status"] = "pass" if ok else "fail"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundles", required=True, help="directory of case bundles")
    parser.add_argument("--report", required=True, help="output report JSON path")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device for the oracle computation (default: cuda if available)",
    )
    args = parser.parse_args()

    report = {
        "torch_version": torch.__version__,
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"
        ),
        "results": [],
    }

    bundle_dirs = sorted(
        d
        for d in os.listdir(args.bundles)
        if os.path.isfile(os.path.join(args.bundles, d, "manifest.json"))
    )
    if not bundle_dirs:
        print(f"no bundles found under {args.bundles}", file=sys.stderr)
        sys.exit(1)

    passed = failed = errored = 0
    for name in bundle_dirs:
        bundle_dir = os.path.join(args.bundles, name)
        try:
            result = run_bundle(bundle_dir, args.device)
        except Exception as exc:  # noqa: BLE001 - report and continue
            result = {"op": name, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        report["results"].append(result)
        status = result["status"]
        if status == "pass":
            passed += 1
        elif status == "fail":
            failed += 1
        else:
            errored += 1
        fwd = result.get("forward", {})
        print(
            f"{status.upper():5s} {result['op']:20s} "
            f"fwd max_abs={fwd.get('max_abs', float('nan')):.3e} "
            f"max_rel={fwd.get('max_rel', float('nan')):.3e}"
            + (f"  ({result.get('error')})" if status == "error" else "")
        )

    report["passed"] = passed
    report["failed"] = failed
    report["errored"] = errored

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"\n{passed} passed, {failed} failed, {errored} errored -> {args.report}")
    sys.exit(0 if failed == 0 and errored == 0 else 1)


if __name__ == "__main__":
    main()
