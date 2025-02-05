# src/auto_tuner.py
import time
from numba import cuda
from .kernel_generator import dynamic_kernel

@cuda.jit
def kernel_tuning_1(A, B, C, n):
    idx = cuda.grid(1)
    if idx < n:
        C[idx] = A[idx] + B[idx]

@cuda.jit
def kernel_tuning_2(A, B, C, n):
    idx = cuda.grid(1)
    if idx < n:
        C[idx] = A[idx] * B[idx]

def benchmark_kernel(kernel, A, B):
    n = len(A)
    C = np.zeros_like(A)
    
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block
    
    A_device = cuda.to_device(A)
    B_device = cuda.to_device(B)
    C_device = cuda.to_device(C)
    
    start = time.time()
    kernel[blocks_per_grid, threads_per_block](A_device, B_device, C_device, n)
    C_device.copy_to_host(C)
    end = time.time()
    
    return end - start

def auto_tune(A, B):
    time_1 = benchmark_kernel(kernel_tuning_1, A, B)
    time_2 = benchmark_kernel(kernel_tuning_2, A, B)
    
    print(f"Kernel 1 Time: {time_1:.4f} seconds")
    print(f"Kernel 2 Time: {time_2:.4f} seconds")
    
    return kernel_tuning_1 if time_1 < time_2 else kernel_tuning_2

