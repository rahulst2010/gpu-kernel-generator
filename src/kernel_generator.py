# src/kernel_generator.py
import numpy as np
from numba import cuda
from .auto_tuner import auto_tune
from .llvm_optimizer import llvm_optimize
from .layer_fusion import fused_layer

@cuda.jit
def dynamic_kernel(A, B, C, n):
    idx = cuda.grid(1)
    if idx < n:
        C[idx] = A[idx] + B[idx]  # Element-wise addition (can be modified as needed)

def generate_dynamic_kernel(A, B, use_fusion=False):
    n = len(A)
    C = np.zeros_like(A)
    
    # Auto-tune and select the best kernel
    best_kernel = auto_tune(A, B)
    
    if use_fusion:
        # Use fused layer if enabled
        C = fused_layer(A, B, C)
    else:
        # Use selected best kernel
        dynamic_kernel(A, B, C, n)
    
    return C


