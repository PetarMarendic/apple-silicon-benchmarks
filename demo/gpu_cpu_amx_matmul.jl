# Demo that will stress each of the compute engines on Apple Silicon: GPU, CPU, AMX
# This is the verbatim code shown in the video: https://youtu.be/HX1B0tlODvY
# N = 16384 will place a significant load on DRAM and was chose to push the system towards high power usage. 
# You may play with smaller values to find the optimal compute throughput of your machine

using LinearAlgebra, BenchmarkTools, Metal

N = 16384
A = rand(Float32, N, N); B = rand(Float32, N, N); C = similar(A)

# GPU
a = MtlArray(A); b = MtlArray(B); c = MtlArray(C)
@benchmark mul!($c, $a, $b)

# CPU
@benchmark mul!($C, $A, $B)

# AMX
using AppleAccelerate
@benchmark mul!($C, $A, $B)
