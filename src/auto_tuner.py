import numpy as np
import subprocess
import time
from src.kernel_generator import KernelGenerator

class AutoTuner:
    def __init__(self, kernel_type="gemm"):
        self.kernel_type = kernel_type
        self.block_sizes = [8, 16, 32]  # Experimenting with different block sizes

    def benchmark_kernel(self, M, N, K):
        best_time = float("inf")
        best_block = None

        for block_size in self.block_sizes:
            env = {"BLOCK_SIZE": str(block_size)}
            start = time.time()
            subprocess.run(["hipcc", "-DBLOCK_SIZE=" + str(block_size), "-o", "temp_kernel", f"src/kernels/{self.kernel_type}_kernel.hip.cpp"], env=env)
            end = time.time()

            elapsed_time = end - start
            if elapsed_time < best_time:
                best_time = elapsed_time
                best_block = block_size

            print(f"Block Size {block_size}: {elapsed_time:.4f}s")

        print(f"Best Block Size: {best_block}")
        return best_block

    def run_optimized_gemm(self, M, N, K):
        best_block = self.benchmark_kernel(M, N, K)
        kernel_gen = KernelGenerator()
        return kernel_gen.run_gemm(np.random.rand(M, K), np.random.rand(K, N), M, N, K)

if __name__ == "__main__":
    tuner = AutoTuner("gemm")
    optimized_result = tuner.run_optimized_gemm(128, 128, 128)
    print("Optimized GEMM Run Complete!")
