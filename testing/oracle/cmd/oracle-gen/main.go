// Command oracle-gen generates PyTorch-oracle case bundles for every
// gradcheck registry op (zerfoo docs/adr/091, plan T1.3):
//
//	go run github.com/zerfoo/ztensor/testing/oracle/cmd/oracle-gen -out /tmp/oracle-bundles
//
// The bundles are then rsynced to the DGX and replayed in torch by
// scripts/oracle/run_oracle.py inside nvcr.io/nvidia/pytorch:26.02-py3 via a
// Spark pod; see scripts/oracle/README.md for the full procedure.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/zerfoo/ztensor/testing/oracle"
)

func main() {
	out := flag.String("out", "", "output directory for the case bundles (required)")
	flag.Parse()
	if *out == "" {
		fmt.Fprintln(os.Stderr, "usage: oracle-gen -out <dir>")
		os.Exit(2)
	}
	sum, err := oracle.GenerateAll(context.Background(), *out)
	if err != nil {
		fmt.Fprintf(os.Stderr, "oracle-gen: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %d bundles to %s\n", len(sum.Written), *out)
	for _, op := range sum.Written {
		fmt.Printf("  bundle  %s\n", op)
	}
	for _, s := range sum.Skipped {
		fmt.Printf("  SKIPPED %s: %s\n", s.Op, s.Reason)
	}
}
