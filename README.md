# Apple Silicon Performance Benchmarks

A collection of standalone scripts for measuring key performance characteristics of Apple M-series systems, including **memory bandwidth**, **compute throughput**, and **cache behavior**.

Each script is self-contained, reproducible, and designed to be easy to run on your own machine.

---

## Contents

| Script | Description |
|--------|------------|
| `julia_roofline_benchmark.jl` | Builds an empirical roofline model (CPU & GPU bandwidth + compute) |

## Quick Start

Run any script directly with Julia:

```bash
julia julia_roofline_benchmark.jl
```

Each script prints results directly to the terminal.

---

## 📚 Documentation

Detailed explanations for each benchmark:

- [Roofline Benchmark](docs/roofline.md)

## Methodology

These benchmarks are designed around simple, well-understood kernels:

**AXPY (`y = αx + y`)** → memory-bound workload
**GEMM (`C = A × B`)** → compute-bound workload

By combining these, we can characterize:
- Peak **memory bandwidth**
- Peak **compute throughput**
- The **performance envelope** of a system

## ⚠️ Notes

- Scripts may allocate **large arrays** to avoid cache effects.
- Results depend on:
  - CPU architecture
  - Memory configuration
  - Number of cores / threads

---

## 🤝 Contributing Results

If you run these benchmarks, feel free to share:

- System specs (CPU, GPU, memory)
- Measured bandwidth (GB/s)
- Compute throughput (GFLOP/s)

This helps build a cross-platform dataset for comparison.

---

## 📄 License

MIT 

