#include <hip/hip_runtime.h>

#define BLOCK_SIZE 16  // Optimized block size for shared memory usage

__global__ void gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < M && t * BLOCK_SIZE + threadIdx.x < K) {
            Asub[threadIdx.y][threadIdx.x] = A[row * K + t * BLOCK_SIZE + threadIdx.x];
        } else {
            Asub[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * BLOCK_SIZE + threadIdx.y < K) {
            Bsub[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * N + col];
        } else {
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
