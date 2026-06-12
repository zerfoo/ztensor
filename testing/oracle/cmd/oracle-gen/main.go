// Command oracle-gen generates PyTorch-oracle case bundles for every
// gradcheck registry op (zerfoo docs/adr/091, plan T1.3):
//
//	go run github.com/zerfoo/ztensor/testing/oracle/cmd/oracle-gen -out /tmp/oracle-bundles
//
// With -engine gpu the bundles record the CUDA GPU engine instead -- the
// oracle gate for the kernel-numerics work (zerfoo plan T3.x). That variant
// requires CUDA and runs on the DGX GB10 inside a Spark pod.
//
// The bundles are then replayed in torch by scripts/oracle/run_oracle.py
// inside nvcr.io/nvidia/pytorch:26.02-py3 via a Spark pod; see
// scripts/oracle/README.md for the full procedure.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/testing/oracle"
)

func main() {
	out := flag.String("out", "", "output directory for the case bundles (required)")
	engine := flag.String("engine", "cpu", "recording engine: cpu | gpu (gpu requires CUDA; run on the GB10 via Spark)")
	flag.Parse()
	if *out == "" {
		fmt.Fprintln(os.Stderr, "usage: oracle-gen -out <dir> [-engine cpu|gpu]")
		os.Exit(2)
	}

	var ec oracle.EngineConfig
	switch *engine {
	case "cpu":
		ec = oracle.EngineConfig{
			Name:   "cpu-f32",
			Engine: compute.NewCPUEngine[float32](numeric.Float32Ops{}),
		}
	case "gpu":
		gpuEng, err := compute.NewGPUEngine[float32](numeric.Float32Ops{})
		if err != nil {
			fmt.Fprintf(os.Stderr, "oracle-gen: NewGPUEngine: %v\n", err)
			os.Exit(1)
		}
		defer func() { _ = gpuEng.Close() }()
		ec = oracle.EngineConfig{
			Name:   "gpu-f32",
			Engine: gpuEng,
			Reset:  gpuEng.ResetPool,
		}
	default:
		fmt.Fprintf(os.Stderr, "oracle-gen: unknown -engine %q (want cpu or gpu)\n", *engine)
		os.Exit(2)
	}

	sum, err := oracle.GenerateAllWith(context.Background(), *out, ec)
	if err != nil {
		fmt.Fprintf(os.Stderr, "oracle-gen: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %d bundles (engine %s) to %s\n", len(sum.Written), sum.Engine, *out)
	for _, op := range sum.Written {
		fmt.Printf("  bundle  %s\n", op)
	}
	for _, s := range sum.Skipped {
		fmt.Printf("  SKIPPED %s: %s\n", s.Op, s.Reason)
	}
}
