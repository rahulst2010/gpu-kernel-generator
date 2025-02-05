# src/layer_fusion.py
from numba import cuda
import numpy as np

@cuda.jit
def fused_layer_kernel(A, B, C, n):
    idx = cuda.grid(1)
    if idx < n:
        C[idx] = A[idx] * B[idx] + C[idx]  # Fusion of multiply and add operations

def fused_layer(A, B, C):
    n = len(A)
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block
    
    A_device = cuda.to_device(A)
    B_device = cuda.to_device(B)
    C_device = cuda.to_device(C)
    
    fused_layer_kernel[blocks_per_grid, threads_per_block](A_device, B_device, C_device, n)
    C_device.copy_to_host(C)
    return C

