import ctypes
import numpy as np
import os

# Load HIP shared libraries
hip_lib = ctypes.cdll.LoadLibrary(os.path.join(os.getcwd(), "src/kernels/libgpu_kernels.so"))

class KernelGenerator:
    def __init__(self):
        self.lib = hip_lib

    def run_gemm(self, A, B, M, N, K):
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.lib.gemm_kernel(A_ptr, B_ptr, C_ptr, M, N, K)
        return C

    def run_conv2d(self, input_tensor, kernel, width, height):
        input_tensor = input_tensor.astype(np.float32)
        kernel = kernel.astype(np.float32)
        output_tensor = np.zeros((height, width), dtype=np.float32)

        input_ptr = input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        kernel_ptr = kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.lib.conv2d_kernel(input_ptr, kernel_ptr, output_ptr, width, height)
        return output_tensor
