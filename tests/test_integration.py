# tests/test_integration.py
import unittest
import numpy as np
from src.kernel_generator import generate_dynamic_kernel
from src.auto_tuner import auto_tune
from src.layer_fusion import fused_layer

class TestIntegration(unittest.TestCase):
    def test_full_integration(self):
        A = np.random.rand(10000).astype(np.float32)
        B = np.random.rand(10000).astype(np.float32)

        # Full flow from auto-tuning to fusion
        C = generate_dynamic_kernel(A, B, use_fusion=True)
        self.assertEqual(C.shape, A.shape)
        self.assertTrue(np.allclose(C, A * B + B))  # Ensure fusion worked
        
        # Test with no fusion
        C = generate_dynamic_kernel(A, B)
        self.assertEqual(C.shape, A.shape)
        self.assertTrue(np.allclose(C, A + B))  # Ensure addition worked correctly

if __name__ == "__main__":
    unittest.main()

