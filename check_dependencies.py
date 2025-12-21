#!/usr/bin/env python3
"""
ABSA项目依赖检查脚本
检查所有必需的依赖包是否正确安装
"""

import sys
import importlib
from typing import List, Tuple

def check_dependency(name: str, package_name: str = None, min_version: str = None) -> Tuple[bool, str]:
    """
    检查单个依赖包

    Args:
        name: 包名
        package_name: 实际导入名（如果与包名不同）
        min_version: 最低版本要求

    Returns:
        (成功标志, 版本信息)
    """
    if package_name is None:
        package_name = name

    try:
        module = importlib.import_module(package_name)

        # 获取版本信息
        version = getattr(module, '__version__', 'unknown')

        # 检查版本要求
        if min_version and hasattr(module, '__version__'):
            from packaging import version
            if version.parse(version) < version.parse(min_version):
                return False, f"{name} {version} (需要 >= {min_version})"

        return True, f"{name} {version}"

    except ImportError as e:
        return False, f"{name} 未安装"
    except Exception as e:
        return False, f"{name} 导入失败: {e}"

def check_all_dependencies():
    """检查所有依赖包"""
    print("=" * 60)
    print("ABSA项目依赖检查")
    print("=" * 60)

    # 定义依赖检查列表
    dependencies = [
        # 核心机器学习框架
        ("torch", "torch"),
        ("torch-geometric", "torch_geometric"),

        # NLP处理
        ("transformers", "transformers"),
        ("stanza", "stanza"),

        # 数值计算和科学计算
        ("numpy", "numpy"),
        ("scipy", "scipy"),

        # 数据处理
        ("pandas", "pandas"),

        # 机器学习和评估
        ("scikit-learn", "sklearn"),

        # 可视化
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),

        # 工具库
        ("lxml", "lxml"),
        ("tqdm", "tqdm"),
    ]

    # 可选依赖
    optional_deps = [
        ("torchtext", "torchtext"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
    ]

    success_count = 0
    total_count = len(dependencies)

    print("检查必需依赖:")
    print("-" * 40)

    for name, package_name in dependencies:
        success, info = check_dependency(name, package_name)
        status = "[OK]" if success else "[FAIL]"
        print("30")
        if success:
            success_count += 1

    print(f"\n必需依赖: {success_count}/{total_count} 个包正常")

    print("\n检查可选依赖:")
    print("-" * 40)

    optional_success = 0
    for name, package_name in optional_deps:
        success, info = check_dependency(name, package_name)
        if success:
            print("30")
            optional_success += 1
        else:
            print("30")

    print(f"\n可选依赖: {optional_success}/{len(optional_deps)} 个包可用")

    print("\n" + "=" * 60)

    if success_count == total_count:
        print("[SUCCESS] All required dependencies are correctly installed!")
        print("\nYou can run the following command to start training:")
        print("python train_simplified.py --model simplified_causal_hafe --dataset semeval2014")
    else:
        print("[WARNING] Some dependencies are missing or incompatible")
        print("\nPlease run the following command to install dependencies:")
        print("pip install -r requirements.txt")
        print("\nOr install missing packages individually")

    print("=" * 60)
    return success_count == total_count

def check_system_info():
    """检查系统信息"""
    print("\n系统信息:")
    print("-" * 20)

    # Python版本
    print(f"Python版本: {sys.version}")

    # 检查CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        gpu_count = torch.cuda.device_count() if cuda_available else 0

        print(f"CUDA可用: {cuda_available}")
        print(f"CUDA版本: {cuda_version}")
        print(f"GPU数量: {gpu_count}")

        if cuda_available:
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(".1f")
    except:
        print("PyTorch CUDA检查失败")

if __name__ == "__main__":
    check_system_info()
    success = check_all_dependencies()
    sys.exit(0 if success else 1)
