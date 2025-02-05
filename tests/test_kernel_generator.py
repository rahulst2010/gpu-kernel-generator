import numpy as np
from src.kernel_generator import KernelGenerator

def test_gemm():
    M, N, K = 4, 4, 4
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)

    kg = KernelGenerator()
    C = kg.run_gemm(A, B, M, N, K)

    expected = np.dot(A, B)
    assert np.allclose(C, expected, atol=1e-3), "GEMM test failed!"

def test_conv2d():
    width, height = 5, 5
    input_tensor = np.random.rand(height, width).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32) / 9  # Simple averaging kernel

    kg = KernelGenerator()
    output = kg.run_conv2d(input_tensor, kernel, width, height)

    expected = np.convolve(input_tensor.flatten(), kernel.flatten(), mode='same').reshape(height, width)
    assert np.allclose(output, expected, atol=1e-3), "Conv2D test failed!"

if __name__ == "__main__":
    test_gemm()
    test_conv2d()
    print("All tests passed!")
