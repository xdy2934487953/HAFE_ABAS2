"""
实验管理和对比工具
用于分析和对比多个ABSA实验的结果
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 添加src到路径
sys.path.append('./src')
from utils import ABSAResultsManager

def list_experiments(experiments_dir="./experiments"):
    """列出所有实验"""
    if not os.path.exists(experiments_dir):
        print(f"实验目录不存在: {experiments_dir}")
        return []

    experiments = []
    for item in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, item)
        if os.path.isdir(exp_path):
            experiments.append(exp_path)

    return sorted(experiments)

def analyze_experiment(exp_dir):
    """分析单个实验"""
    print(f"\n分析实验: {os.path.basename(exp_dir)}")

    # 加载配置
    config_file = os.path.join(exp_dir, "configs", "experiment_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"  模型: {config.get('model', 'N/A')}")
        print(f"  数据集: {config.get('dataset', 'N/A')}")
        print(f"  学习率: {config.get('lr', 'N/A')}")

    # 加载报告
    report_file = os.path.join(exp_dir, "experiment_report.json")
    if os.path.exists(report_file):
        with open(report_file, 'r') as f:
            report = json.load(f)

        best_metrics = report.get('best_metrics', {})
        if best_metrics:
            print(f"  最佳Macro-F1: {best_metrics.get('macro_f1', 0):.4f}")
            print(f"  最佳Accuracy: {best_metrics.get('accuracy', 0):.4f}")
            print(f"  取得最佳的轮数: {best_metrics.get('epoch', 'N/A')}")

    # 检查文件
    files = {
        "训练日志": "logs/train_log.csv",
        "评估日志": "logs/eval_log.csv",
        "最佳模型": "checkpoints/best.pt",
        "最新模型": "checkpoints/latest.pt",
        "训练曲线": "plots/training_curves.png",
        "配置": "configs/experiment_config.json",
        "报告": "experiment_report.json"
    }

    print("  文件状态:")
    for name, path in files.items():
        full_path = os.path.join(exp_dir, path)
        status = "✓" if os.path.exists(full_path) else "✗"
        print(f"    {status} {name}")

def compare_experiments(exp_dirs, output_file=None):
    """对比多个实验"""
    manager = ABSAResultsManager()

    # 加载实验
    exp_names = []
    for exp_dir in exp_dirs:
        exp_name = manager.load_experiment(exp_dir)
        exp_names.append(exp_name)

    if not exp_names:
        print("没有成功加载任何实验")
        return

    # 生成对比报告
    comparison = manager.generate_comparison_report(exp_names, output_file)

    print(f"\n对比结果 ({len(exp_names)} 个实验):")
    print("=" * 60)

    summary = comparison.get('summary', {})
    if summary:
        best_exp = summary.get('best_experiment', 'N/A')
        best_f1 = summary.get('best_macro_f1', 0)
        avg_f1 = summary.get('avg_macro_f1', 0)

        print(f"最佳实验: {best_exp}")
        print(f"最佳Macro-F1: {best_f1:.4f}")
        print(f"平均Macro-F1: {avg_f1:.4f}")

    # 详细对比
    experiments = comparison.get('experiments', {})
    if experiments:
        print(f"\n详细结果:")
        print("-" * 60)
        print(f"{'Experiment':<35} {'Macro-F1':<10} {'Accuracy':<10}")
        print("-" * 60)

        for exp_name, exp_data in experiments.items():
            config = exp_data.get('config', {})
            best_metrics = exp_data.get('best_metrics', {})

            model = config.get('model', 'N/A')
            macro_f1 = best_metrics.get('macro_f1', 0)
            accuracy = best_metrics.get('accuracy', 0)

            print(f"{exp_name:<35} {macro_f1:.4f}    {accuracy:.4f}")

    if output_file:
        print(f"\n对比报告已保存到: {output_file}")

def plot_comparison(exp_dirs, metrics=['macro_f1', 'accuracy'], save_path=None):
    """绘制实验对比图"""
    manager = ABSAResultsManager()

    # 加载实验
    results = {}
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        manager.load_experiment(exp_dir)
        exp_data = manager.experiments[exp_name]

        if exp_data['eval_history']:
            results[exp_name] = {
                'config': exp_data['config'],
                'history': exp_data['eval_history']
            }

    if not results:
        print("没有可用的实验数据进行绘图")
        return

    # 创建图表
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for exp_name, exp_data in results.items():
            history = exp_data['history']
            epochs = [h['epoch'] for h in history]
            values = [h.get(metric, 0) for h in history]

            ax.plot(epochs, values, label=exp_name, marker='o', linewidth=2, markersize=4)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    else:
        plt.show()

    plt.close()

def create_summary_table(exp_dirs, output_file=None):
    """创建实验汇总表格"""
    manager = ABSAResultsManager()

    summary_data = []
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)

        # 加载实验
        manager.load_experiment(exp_dir)
        exp_data = manager.experiments.get(exp_name, {})

        config = exp_data.get('config', {})
        eval_history = exp_data.get('eval_history', [])

        if eval_history:
            # 找到最佳结果
            best_result = max(eval_history, key=lambda x: x.get('macro_f1', 0))

            row = {
                'Experiment': exp_name,
                'Model': config.get('model', 'N/A'),
                'Dataset': config.get('dataset', 'N/A'),
                'Learning_Rate': config.get('lr', 'N/A'),
                'Hidden_Dim': config.get('hidden_dim', 'N/A'),
                'Best_Epoch': best_result.get('epoch', 'N/A'),
                'Best_Macro_F1': f"{best_result.get('macro_f1', 0):.4f}",
                'Best_Accuracy': f"{best_result.get('accuracy', 0):.4f}",
                'Final_Macro_F1': f"{eval_history[-1].get('macro_f1', 0):.4f}",
                'Final_Accuracy': f"{eval_history[-1].get('accuracy', 0):.4f}"
            }

            if 'gini' in best_result:
                row['Gini'] = f"{best_result.get('gini', 0):.4f}"
                row['DP_Aspect'] = f"{best_result.get('dp_aspect', 0):.4f}"

            summary_data.append(row)

    if not summary_data:
        print("没有可用的实验数据")
        return

    # 创建DataFrame
    df = pd.DataFrame(summary_data)

    # 按最佳Macro-F1排序
    df = df.sort_values('Best_Macro_F1', ascending=False)

    print("\n实验汇总表格:")
    print("=" * 100)
    print(df.to_string(index=False))

    if output_file:
        if output_file.endswith('.csv'):
            df.to_csv(output_file, index=False)
        elif output_file.endswith('.json'):
            df.to_dict('records')
            with open(output_file, 'w') as f:
                json.dump(df.to_dict('records'), f, indent=4)
        else:
            df.to_csv(output_file, index=False)

        print(f"\n汇总表格已保存到: {output_file}")

    return df

def main():
    parser = argparse.ArgumentParser(description='ABSA实验管理和对比工具')
    parser.add_argument('--action', type=str, required=True,
                       choices=['list', 'analyze', 'compare', 'plot', 'table'],
                       help='执行的操作')
    parser.add_argument('--experiments', nargs='+',
                       help='实验目录列表（用于compare, plot, table）')
    parser.add_argument('--experiments_dir', type=str, default='./experiments',
                       help='实验根目录')
    parser.add_argument('--output', type=str,
                       help='输出文件路径')
    parser.add_argument('--metrics', nargs='+', default=['macro_f1', 'accuracy'],
                       help='要对比的指标（用于plot）')

    args = parser.parse_args()

    # 获取实验列表
    if args.action == 'list':
        experiments = list_experiments(args.experiments_dir)
        print(f"找到 {len(experiments)} 个实验:")
        for exp in experiments:
            print(f"  {os.path.basename(exp)}")

    elif args.action == 'analyze':
        experiments = list_experiments(args.experiments_dir)
        if not experiments:
            print("没有找到实验")
            return

        for exp in experiments:
            analyze_experiment(exp)

    elif args.action in ['compare', 'plot', 'table']:
        if not args.experiments:
            # 如果没有指定实验，使用所有实验
            args.experiments = list_experiments(args.experiments_dir)

        if not args.experiments:
            print("没有找到实验")
            return

        if args.action == 'compare':
            compare_experiments(args.experiments, args.output)
        elif args.action == 'plot':
            plot_comparison(args.experiments, args.metrics, args.output)
        elif args.action == 'table':
            create_summary_table(args.experiments, args.output)

if __name__ == "__main__":
    main()
