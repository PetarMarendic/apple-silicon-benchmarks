# Roofline Benchmark

This script builds an **empirical roofline model** of your system by measuring two fundamental limits:

- **Memory bandwidth** (GB/s)
- **Compute throughput** (GFLOP/s)

It does this using two simple, well-understood kernels:
- **AXPY** → memory-bound
- **GEMM (matrix multiplication)** → compute-bound

---

## What This Measures

The script reports:

### CPU
- Single-core memory bandwidth
- Multi-core memory bandwidth
- Peak compute throughput

### GPU (Metal)
- Memory bandwidth
- Compute throughput


---
## 🧠 Methodology

### 1. Memory Bandwidth — AXPY

We use the operation:

```
y = αx + y
```

Per element:
- 2 FLOPs (1 multiply + 1 add)
- 3 memory operations:
  - read `x`
  - read `y`
  - write `y`

For `Float32`:
- 4 bytes per element
- total traffic = **12 bytes per element**

So bandwidth is computed as:

```
bandwidth = (N × 12 bytes) / time
```

---

### 2. Compute Throughput — GEMM

We use:

```
C = A × B
```

For an `N × N` matrix multiply:
- total FLOPs ≈ **2N³**

Compute throughput is:

```
GFLOPs = (2 × N³) / time
```

Matrix multiplication has **high arithmetic intensity**, meaning:
- data is reused many times
- performance is limited by compute, not memory

---

## 🚀 Usage

Run the script:

```
julia julia_roofline_benchmark.jl
```

It will automatically execute:
- CPU benchmarks
- GPU benchmarks

---


## ⚠️ Important Notes

### Large Allocations
- Vectors: up to 2³⁰ elements
- Matrices: up to 16384 × 16384

These sizes are intentional:
- avoid cache effects
- force sustained DRAM bandwidth

---


