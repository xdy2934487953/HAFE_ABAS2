"""
快速测试简化版Causal-HAFE
不依赖torch_geometric，仅测试基本功能
"""

import sys
sys.path.append('./src')

def test_imports():
    """测试导入"""
    print("Testing imports...")
    try:
        # 测试基本导入
        import torch
        import torch.nn as nn
        print("[OK] PyTorch available")

        # 测试numpy
        import numpy as np
        print("[OK] NumPy available")

        # 测试sklearn
        from sklearn.metrics import f1_score
        print("[OK] Scikit-learn available")

        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False

def test_dib_module():
    """测试DIB模块"""
    print("\nTesting DIB module...")
    try:
        from disentangled_information_bottleneck import DIBModule, test_dib_module
        test_dib_module()
        print("[OK] DIB module test passed")
        return True
    except Exception as e:
        print(f"[FAIL] DIB test failed: {e}")
        return False

def test_graph_builder():
    """测试图构建器"""
    print("\nTesting graph builder...")
    try:
        from graph_builder import EdgeType, NodeType
        print(f"[OK] Edge types: {len([attr for attr in dir(EdgeType) if not attr.startswith('_')])}")
        print(f"[OK] Node types: {len([attr for attr in dir(NodeType) if not attr.startswith('_')])}")
        return True
    except Exception as e:
        print(f"[FAIL] Graph builder test failed: {e}")
        return False

def test_config():
    """测试训练配置"""
    print("\nTesting training config...")
    try:
        import train_simplified
        print("[OK] Training script imports correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Training script error: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("Simplified Causal-HAFE Quick Test")
    print("=" * 50)

    tests = [
        ("Basic imports", test_imports),
        ("DIB module", test_dib_module),
        ("Graph builder", test_graph_builder),
        ("Training config", test_config),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"[FAIL] {test_name} failed")
        except Exception as e:
            print(f"[ERROR] {test_name} error: {e}")

    print("\n" + "=" * 50)
    print(f"Test results: {passed}/{total} passed")

    if passed == total:
        print("[SUCCESS] All tests passed! Simplified Causal-HAFE is ready.")
        print("\nRun command:")
        print("python train_simplified.py --model simplified_causal_hafe --dataset semeval2014")
    else:
        print("[WARNING] Some tests failed, please check dependencies.")
        print("\nInstall dependencies:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
