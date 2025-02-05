#include <hip/hip_runtime.h>

#define KERNEL_SIZE 3  // Assuming a 3x3 convolution kernel

__global__ void conv2d_kernel(float* input, float* kernel, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int i = -KERNEL_SIZE / 2; i <= KERNEL_SIZE / 2; i++) {
        for (int j = -KERNEL_SIZE / 2; j <= KERNEL_SIZE / 2; j++) {
            int ix = x + i;
            int iy = y + j;
            if (ix >= 0 && iy >= 0 && ix < width && iy < height) {
                sum += input[iy * width + ix] * kernel[(i + KERNEL_SIZE / 2) * KERNEL_SIZE + (j + KERNEL_SIZE / 2)];
            }
        }
    }
    output[y * width + x] = sum;
}
