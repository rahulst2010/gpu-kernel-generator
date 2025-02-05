# tests/test_kernel_generator.py
import unittest
import numpy as np
from src.kernel_generator import generate_dynamic_kernel

class TestKernelGeneration(unittest.TestCase):
    def test_kernel(self):
        A = np.random.rand(10000).astype(np.float32)
        B = np.random.rand(10000).astype(np.float32)
        
        # Test with layer fusion
        C = generate_dynamic_kernel(A, B, use_fusion=True)
        self.assertEqual(C.shape, A.shape)
        self.assertTrue(np.allclose(C, A * B + B))  # Fused operation
        
        # Test without fusion (default)
        C = generate_dynamic_kernel(A, B)
        self.assertEqual(C.shape, A.shape)
        self.assertTrue(np.allclose(C, A + B))  # Element-wise addition

if __name__ == "__main__":
    unittest.main()

