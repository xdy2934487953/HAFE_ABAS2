import re
import os
import sys

def parse_log(log_file):
    """解析日志文件提取关键指标"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        try:
            with open(log_file, 'r', encoding='gbk') as f:
                content = f.read()
        except:
            return {}
    
    results = {}
    
    # 提取指标
    patterns = {
        'Accuracy': r'Accuracy: ([\d.]+)%',
        'Macro-F1': r'Macro-F1: ([\d.]+)%',
        'Gini': r'Gini: ([\d.]+)',
        'Gap': r'Gap: ([\d.]+)',
        'DP-Aspect': r'DP-Aspect: ([\d.]+)',
        'High-Freq F1': r'High-Freq F1: ([\d.]+)',
        'Low-Freq F1': r'Low-Freq F1: ([\d.]+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            results[key] = float(match.group(1))
    
    return results

def print_colored(text, color='white'):
    """Windows彩色输出"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    
    if sys.platform == 'win32':
        # Windows 10+ 支持ANSI颜色
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def main():
    results_dir = './results'
    
    if not os.path.exists(results_dir):
        print_colored("错误：results 目录不存在！", 'red')
        print("请先运行实验生成结果文件")
        return
    
    experiments = [
        ('semeval2014', 'baseline'),
        ('semeval2014', 'hafe'),
        ('semeval2016', 'baseline'),
        ('semeval2016', 'hafe'),
    ]
    
    print_colored("="*80, 'cyan')
    print_colored("HAFE-ABSA 实验结果对比", 'cyan')
    print_colored("="*80, 'cyan')
    
    all_results = {}
    missing_files = []
    
    for dataset, model in experiments:
        log_file = os.path.join(results_dir, f'{dataset}_{model}.log')
        if os.path.exists(log_file):
            results = parse_log(log_file)
            if results:
                all_results[f'{dataset}_{model}'] = results
            else:
                missing_files.append(log_file)
        else:
            missing_files.append(log_file)
    
    if missing_files:
        print_colored("\n警告：以下日志文件不存在或解析失败:", 'yellow')
        for f in missing_files:
            print(f"  - {f}")
        print()
    
    if not all_results:
        print_colored("错误：没有找到任何有效的结果文件！", 'red')
        return
    
    # 打印表格
    print()
    header = f"{'Dataset':<15} {'Model':<10} {'Acc':<8} {'Macro-F1':<10} {'Gini':<8} {'Gap':<8} {'DP':<8}"
    print(header)
    print("-"*80)
    
    for dataset in ['semeval2014', 'semeval2016']:
        for model in ['baseline', 'hafe']:
            key = f'{dataset}_{model}'
            if key in all_results:
                r = all_results[key]
                row = (f"{dataset:<15} {model:<10} "
                      f"{r.get('Accuracy', 0):<8.2f} "
                      f"{r.get('Macro-F1', 0):<10.2f} "
                      f"{r.get('Gini', 0):<8.4f} "
                      f"{r.get('Gap', 0):<8.4f} "
                      f"{r.get('DP-Aspect', 0):<8.4f}")
                
                if model == 'hafe':
                    print_colored(row, 'green')
                else:
                    print(row)
        print("-"*80)
    
    # 计算改进
    print_colored("\nHAFE相对于Baseline的改进:", 'yellow')
    print("-"*80)
    
    for dataset in ['semeval2014', 'semeval2016']:
        baseline_key = f'{dataset}_baseline'
        hafe_key = f'{dataset}_hafe'
        
        if baseline_key in all_results and hafe_key in all_results:
            baseline = all_results[baseline_key]
            hafe = all_results[hafe_key]
            
            print_colored(f"\n{dataset.upper()}:", 'cyan')
            
            # Macro-F1改进
            macro_f1_diff = hafe.get('Macro-F1', 0) - baseline.get('Macro-F1', 0)
            macro_f1_str = f"  Macro-F1: {baseline.get('Macro-F1', 0):.2f} → {hafe.get('Macro-F1', 0):.2f} ({macro_f1_diff:+.2f})"
            if macro_f1_diff > 0:
                print_colored(macro_f1_str, 'green')
            else:
                print(macro_f1_str)
            
            # Gini改进
            if baseline.get('Gini', 0) > 0:
                gini_improve = (baseline.get('Gini', 0) - hafe.get('Gini', 0)) / baseline.get('Gini', 1) * 100
                gini_str = f"  Gini: {baseline.get('Gini', 0):.4f} → {hafe.get('Gini', 0):.4f} ({gini_improve:.1f}% 降低)"
                if gini_improve > 0:
                    print_colored(gini_str, 'green')
                else:
                    print(gini_str)
            
            # 低频F1改进
            low_freq_diff = hafe.get('Low-Freq F1', 0) - baseline.get('Low-Freq F1', 0)
            low_freq_str = f"  Low-Freq F1: {baseline.get('Low-Freq F1', 0):.4f} → {hafe.get('Low-Freq F1', 0):.4f} ({low_freq_diff:+.4f})"
            if low_freq_diff > 0:
                print_colored(low_freq_str, 'green')
            else:
                print(low_freq_str)
    
    print()
    print_colored("="*80, 'cyan')
    print_colored("对比完成！", 'cyan')
    print_colored("="*80, 'cyan')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print_colored(f"\n错误: {e}", 'red')
        import traceback
        traceback.print_exc()
    finally:
        if sys.platform == 'win32':
            input("\n按回车键退出...")