import pytest
from src.auto_tuner import AutoTuner

def test_tuner():
    tuner = AutoTuner("gemm")
    best_block = tuner.benchmark_kernel(128, 128, 128)
    assert best_block in [8, 16, 32], "Auto-tuner failed to find optimal block size!"

if __name__ == "__main__":
    test_tuner()
    print("Auto-Tuner Test Passed!")
