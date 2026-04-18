#Description: script that reproduces the demo from the video Apple Silicon Performance Explained: Bandwidth vs Compute (Hands-On)
#Youtube: https://youtu.be/wV7bK8IhUn4
using LinearAlgebra
using BenchmarkTools
using Metal

# ------------------------------------------------------------
# Empirical roofline demo:
#   1) CPU single-core bandwidth via y .+= x
#   2) CPU multi-core bandwidth via BLAS axpy!
#   3) GPU bandwidth via Metal axpy!
#   4) CPU compute throughput via matrix multiply
#   5) GPU compute throughput via Metal Float32 matrix multiply
#   6) GPU compute throughput via Metal Float16 matrix multiply
# ------------------------------------------------------------

# -------------------------
# Configuration
# -------------------------
const VEC_N = 2^30              # bandwidth test size
const MAT_N = 8192              # compute test size
const ALPHA = 1.0f0
const MAT_SECONDS = 5           # benchmarking window for GEMM

# -------------------------
# Helpers
# -------------------------
bytes_moved_axpy(n::Integer, ::Type{T}) where {T} = 3n * sizeof(T)
# read x + read y + write y = 3 memory ops per element

bandwidth_gbs(n::Integer, t::Float64, ::Type{T}) where {T} =
    bytes_moved_axpy(n, T) / t / 1e9

gemm_gflops(n::Integer, t::Float64) = 2e-9 * n^3 / t
# Standard GEMM estimate: 2N^3 FLOPs

function trial_min_seconds(trial)
    return minimum(trial).time / 1e9
end

function print_result(label::AbstractString, value::Real, unit::AbstractString)
    println(rpad(label * ":", 34), round(value; digits=3), " ", unit)
end

# -------------------------
# CPU bandwidth
# -------------------------
println("=== CPU bandwidth ===")
println("Allocating vectors...")
x = rand(Float32, VEC_N)
y = rand(Float32, VEC_N)

println("\nBenchmarking single-core broadcast: y .+= x")
trial_cpu_single = @benchmark $y .+= $x
t_cpu_single = trial_min_seconds(trial_cpu_single)
bw_cpu_single = bandwidth_gbs(length(y), t_cpu_single, Float32)

println(trial_cpu_single)
print_result("CPU single-core bandwidth", bw_cpu_single, "GB/s")

println("\nBenchmarking multi-core BLAS AXPY: axpy!(α, y, x)")
trial_cpu_multi = @benchmark axpy!($ALPHA, $y, $x)
t_cpu_multi = trial_min_seconds(trial_cpu_multi)
bw_cpu_multi = bandwidth_gbs(length(y), t_cpu_multi, Float32)

println(trial_cpu_multi)
print_result("CPU multi-core bandwidth", bw_cpu_multi, "GB/s")

# -------------------------
# GPU bandwidth
# -------------------------
println("\n=== GPU bandwidth ===")
println("Copying vectors to Metal...")
x_m = MtlArray(x)
y_m = MtlArray(y)

# Warm-up
axpy!(ALPHA, y_m, x_m)
synchronize()

println("\nBenchmarking GPU AXPY: axpy!(α, y_m, x_m) + synchronize()")
trial_gpu_axpy = @benchmark begin
    axpy!($ALPHA, $y_m, $x_m)
    synchronize()
end
t_gpu_axpy = trial_min_seconds(trial_gpu_axpy)
bw_gpu = bandwidth_gbs(length(y), t_gpu_axpy, Float32)

println(trial_gpu_axpy)
print_result("GPU bandwidth", bw_gpu, "GB/s")

# -------------------------
# CPU compute throughput
# -------------------------
println("\n=== CPU compute throughput ===")
println("Allocating CPU matrices...")
A = rand(Float32, MAT_N, MAT_N)
B = rand(Float32, MAT_N, MAT_N)
C = similar(B)

println("\nBenchmarking CPU GEMM: mul!(C, A, B)")
trial_cpu_gemm = @benchmark mul!($C, $A, $B) seconds=MAT_SECONDS
t_cpu_gemm = trial_min_seconds(trial_cpu_gemm)
gflops_cpu = gemm_gflops(MAT_N, t_cpu_gemm)

println(trial_cpu_gemm)
print_result("CPU compute throughput", gflops_cpu, "GFLOP/s")

# -------------------------
# GPU compute throughput (Float32)
# -------------------------
println("\n=== GPU compute throughput ===")
println("Copying Float32 matrices to Metal...")
A_m = MtlArray(A)
B_m = MtlArray(B)
C_m = MtlArray(C)

# Warm-up
mul!(C_m, A_m, B_m)
synchronize()

println("\nBenchmarking GPU GEMM Float32: mul!(C_m, A_m, B_m) + synchronize()")
trial_gpu_gemm = @benchmark begin
    mul!($C_m, $A_m, $B_m)
    synchronize()
end seconds=MAT_SECONDS
t_gpu_gemm = trial_min_seconds(trial_gpu_gemm)
gflops_gpu = gemm_gflops(MAT_N, t_gpu_gemm)

println(trial_gpu_gemm)
print_result("GPU compute throughput FP32", gflops_gpu, "GFLOP/s")

# -------------------------
# GPU compute throughput (Float16)
# -------------------------
println("\nCopying Float16 matrices to Metal...")
A16 = rand(Float16, MAT_N, MAT_N)
B16 = rand(Float16, MAT_N, MAT_N)
C16 = zeros(Float16, MAT_N, MAT_N)

A16_m = MtlArray(A16)
B16_m = MtlArray(B16)
C16_m = MtlArray(C16)

# Warm-up
mul!(C16_m, A16_m, B16_m)
synchronize()

println("\nBenchmarking GPU GEMM Float16: mul!(C16_m, A16_m, B16_m) + synchronize()")
trial_gpu_gemm_fp16 = @benchmark begin
    mul!($C16_m, $A16_m, $B16_m)
    synchronize()
end seconds=MAT_SECONDS
t_gpu_gemm_fp16 = trial_min_seconds(trial_gpu_gemm_fp16)
gflops_gpu_fp16 = gemm_gflops(MAT_N, t_gpu_gemm_fp16)

println(trial_gpu_gemm_fp16)
print_result("GPU compute throughput FP16", gflops_gpu_fp16, "GFLOP/s")
print_result("GPU FP16 / FP32 throughput", gflops_gpu_fp16 / gflops_gpu, "x")

# -------------------------
# Summary
# -------------------------
println("\n================ SUMMARY ================")
print_result("CPU single-core bandwidth", bw_cpu_single, "GB/s")
print_result("CPU multi-core bandwidth", bw_cpu_multi, "GB/s")
print_result("GPU bandwidth", bw_gpu, "GB/s")
print_result("CPU compute throughput", gflops_cpu, "GFLOP/s")
print_result("GPU compute throughput FP32", gflops_gpu, "GFLOP/s")
print_result("GPU compute throughput FP16", gflops_gpu_fp16, "GFLOP/s")
print_result("GPU FP16 / FP32 throughput", gflops_gpu_fp16 / gflops_gpu, "x")
println("========================================")
