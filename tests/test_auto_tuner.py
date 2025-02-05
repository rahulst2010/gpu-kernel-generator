# tests/test_auto_tuner.py
import unittest
import numpy as np
from src.auto_tuner import auto_tune

class TestAutoTuner(unittest.TestCase):
    def test_auto_tune(self):
        A = np.random.rand(10000).astype(np.float32)
        B = np.random.rand(10000).astype(np.float32)
        
        best_kernel = auto_tune(A, B)
        self.assertIn(best_kernel, [kernel_tuning_1, kernel_tuning_2])

if __name__ == "__main__":
    unittest.main()

