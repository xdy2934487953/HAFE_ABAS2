"""
自动下载Stanza模型到项目本地目录
解决Windows权限问题
"""

import os
import sys

def download_stanza_to_local():
    """下载Stanza模型到项目本地目录"""

    # 获取项目根目录（脚本所在目录）
    project_root = os.path.dirname(os.path.abspath(__file__))
    stanza_dir = os.path.join(project_root, 'stanza_resources')

    print("="*60)
    print("Stanza模型自动下载脚本")
    print("="*60)
    print(f"项目目录: {project_root}")
    print(f"下载目标: {stanza_dir}")
    print("="*60)

    # 创建目录
    os.makedirs(stanza_dir, exist_ok=True)

    # 设置环境变量，强制Stanza使用本地目录
    os.environ['STANZA_RESOURCES_DIR'] = stanza_dir

    # 切换工作目录到stanza_resources
    original_dir = os.getcwd()

    try:
        import stanza

        print("\n开始下载Stanza英文模型...")
        print("这可能需要5-10分钟，请耐心等待...")
        print("下载内容约400-500MB\n")

        # 下载模型
        stanza.download(
            lang='en',
            dir=stanza_dir,
            model_dir=stanza_dir,
            verbose=True,
            logging_level='INFO'
        )

        print("\n" + "="*60)
        print("✅ 下载完成!")
        print("="*60)
        print(f"模型位置: {stanza_dir}")
        print("\n现在可以运行训练脚本了:")
        print("  python train_causal.py --dataset semeval2014 --model causal_hafe --epochs 10")
        print("="*60)

        return True

    except ImportError:
        print("\n❌ 错误: 未找到stanza库")
        print("请先安装: pip install stanza")
        return False

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的原因:")
        print("1. 网络连接问题")
        print("2. 磁盘空间不足（需要至少1GB）")
        print("3. 防火墙阻止下载")
        print("\n解决方案:")
        print("- 检查网络连接")
        print("- 使用VPN（如果在国内）")
        print("- 关闭防火墙/杀毒软件重试")
        return False

    finally:
        os.chdir(original_dir)

if __name__ == '__main__':
    success = download_stanza_to_local()

    if not success:
        print("\n如果多次失败，请联系我使用备用方案:")
        print("1. 使用SpaCy替代Stanza（更简单）")
        print("2. 手动下载离线安装包")
        sys.exit(1)
