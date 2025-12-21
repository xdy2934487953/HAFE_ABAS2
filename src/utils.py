import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import scipy.sparse as sp
from scipy.spatial import distance_matrix
import pandas as pd
import os
from torch.autograd import grad
import json
import csv
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# 自动检测设备：如果是 Mac M芯片用 mps，如果是 NVIDIA 用 cuda，否则用 cpu
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map = np.array(idx_map)
    return idx_map


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


class Results:
    def __init__(self, seed_num, model_num, args):
        super(Results, self).__init__()

        self.seed_num = seed_num
        self.model_num = model_num
        self.dataset = args.dataset
        self.model = args.model
        self.auc, self.f1, self.acc, self.parity, self.equality = np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num))

    def report_results(self):
        for i in range(self.model_num):
            print(f"============" + f"{self.dataset}" + "+" + f"{self.model}" + "============")
            print(f"AUCROC: {np.around(np.mean(self.auc[:, i]) * 100, 2)} ± {np.around(np.std(self.auc[:, i]) * 100, 2)}")
            print(f'F1-score: {np.around(np.mean(self.f1[:, i]) * 100, 2)} ± {np.around(np.std(self.f1[:, i]) * 100, 2)}')
            print(f'ACC: {np.around(np.mean(self.acc[:, i]) * 100, 2)} ± {np.around(np.std(self.acc[:, i]) * 100, 2)}')
            print(f'Parity: {np.around(np.mean(self.parity[:, i]) * 100, 2)} ± {np.around(np.std(self.parity[:, i]) * 100, 2)}')
            print(f'Equality: {np.around(np.mean(self.equality[:, i]) * 100, 2)} ± {np.around(np.std(self.equality[:, i]) * 100, 2)}')
            print("=================END=================")

    def save_results(self, args):
        for i in range(self.model_num):
            # 构建保存路径
            save_path = os.path.join(args.log_dir, f"{args.dataset}_{args.encoder}.json")

            # 确保目录存在
            dir_path = os.path.dirname(save_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            # 将args转换为字典并删除不需要的键
            args_dict = vars(args)
            del args_dict['device']
            if 'pbar' in args_dict:
                del args_dict['pbar']

            # 写入args_dict到文件
            with open(save_path, 'w') as file:
                json.dump(args_dict, file, indent=4)
                file.write('\n')

            # 计算并写入性能指标
            with open(save_path, 'a') as file:
                ret_dict = {
                    "AUC": f"{np.around(np.mean(self.auc[:, i]) * 100, 2)} ± {np.around(np.std(self.auc[:, i]) * 100, 2)}",
                    "F1": f"{np.around(np.mean(self.f1[:, i]) * 100, 2)} ± {np.around(np.std(self.f1[:, i]) * 100, 2)}",
                    "ACC": f"{np.around(np.mean(self.acc[:, i]) * 100, 2)} ± {np.around(np.std(self.acc[:, i]) * 100, 2)}",
                    'Parity': f'{np.around(np.mean(self.parity[:, i]) * 100, 2)} ± {np.around(np.std(self.parity[:, i]) * 100, 2)}',
                    'Equality': f'{np.around(np.mean(self.equality[:, i]) * 100, 2)} ± {np.around(np.std(self.equality[:, i]) * 100, 2)}'
                }
                json.dump(ret_dict, file, indent=4, ensure_ascii=False)

            # 构建ret_dict的name键
            ret_dict[
                'name'] = "FairINV_" + args.dataset + f"_{args.encoder}" + f"_alpha:{args.alpha}" + f"_lr_sp:{args.lr_sp}" + f"_env_num:{args.env_num}" + f"_lr:{args.lr}"

            # 写入results.json文件
            with open('results.json', 'a') as file:
                json.dump(ret_dict, file, indent=4, ensure_ascii=False)
                file.write('\n')


# ===== ABSA训练结果保存系统 =====

class ExperimentLogger:
    """
    ABSA实验日志记录器
    支持训练过程、模型保存、配置管理、可视化等功能
    """

    def __init__(self, experiment_name, save_dir="./experiments", config=None):
        """
        Args:
            experiment_name: 实验名称
            save_dir: 保存目录
            config: 实验配置字典
        """
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(save_dir, f"{experiment_name}_{self.timestamp}")

        # 创建目录结构
        self._create_directories()

        # 保存配置
        self.config = config or {}
        self.save_config()

        # 初始化日志文件
        self.train_log_path = os.path.join(self.experiment_dir, "logs", "train_log.csv")
        self.eval_log_path = os.path.join(self.experiment_dir, "logs", "eval_log.csv")
        self._init_log_files()

        # 训练历史
        self.train_history = []
        self.eval_history = []
        self.best_metrics = {}

    def _create_directories(self):
        """创建实验目录结构"""
        subdirs = ["checkpoints", "logs", "plots", "configs"]
        for subdir in subdirs:
            os.makedirs(os.path.join(self.experiment_dir, subdir), exist_ok=True)

    def _init_log_files(self):
        """初始化日志文件"""
        # 训练日志头
        train_headers = ["epoch", "train_loss", "task_loss", "indep_loss", "bias_loss", "ib_loss",
                        "lr", "grad_norm", "time_elapsed"]
        with open(self.train_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(train_headers)

        # 评估日志头
        eval_headers = ["epoch", "accuracy", "macro_f1", "micro_f1", "gini", "dp_aspect",
                       "high_freq_f1", "low_freq_f1", "variance", "gap"]
        with open(self.eval_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(eval_headers)

    def save_config(self, config=None):
        """保存实验配置"""
        if config:
            self.config.update(config)

        config_path = os.path.join(self.experiment_dir, "configs", "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4, default=str)

    def log_train_step(self, epoch, loss_dict, lr=None, grad_norm=None, time_elapsed=None):
        """记录训练步骤"""
        row = [
            epoch,
            loss_dict.get('total', 0),
            loss_dict.get('task', 0),
            loss_dict.get('indep', 0),
            loss_dict.get('bias', 0),
            loss_dict.get('ib', 0),
            lr or 0,
            grad_norm or 0,
            time_elapsed or 0
        ]

        with open(self.train_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # 保存到历史
        self.train_history.append({
            'epoch': epoch,
            **loss_dict,
            'lr': lr,
            'grad_norm': grad_norm,
            'time_elapsed': time_elapsed
        })

    def log_eval_step(self, epoch, metrics):
        """记录评估步骤"""
        row = [
            epoch,
            metrics.get('accuracy', 0),
            metrics.get('macro_f1', 0),
            metrics.get('micro_f1', 0),
            metrics.get('gini', 0),
            metrics.get('dp_aspect', 0),
            metrics.get('high_freq_f1', 0),
            metrics.get('low_freq_f1', 0),
            metrics.get('variance', 0),
            metrics.get('gap', 0)
        ]

        with open(self.eval_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # 保存到历史
        self.eval_history.append({
            'epoch': epoch,
            **metrics
        })

        # 更新最佳指标
        if not self.best_metrics or metrics.get('macro_f1', 0) > self.best_metrics.get('macro_f1', 0):
            self.best_metrics = metrics.copy()
            self.best_metrics['epoch'] = epoch

    def save_model(self, model, optimizer=None, scheduler=None, epoch=None, is_best=False):
        """保存模型checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'timestamp': self.timestamp
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # 保存最新checkpoint
        latest_path = os.path.join(self.experiment_dir, "checkpoints", "latest.pt")
        torch.save(checkpoint, latest_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.experiment_dir, "checkpoints", "best.pt")
            torch.save(checkpoint, best_path)

        # 定期保存epoch checkpoint
        if epoch and epoch % 10 == 0:
            epoch_path = os.path.join(self.experiment_dir, "checkpoints", f"epoch_{epoch}.pt")
            torch.save(checkpoint, epoch_path)

    def load_model(self, model, checkpoint_path, optimizer=None, scheduler=None):
        """加载模型checkpoint"""
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        return epoch

    def plot_training_curves(self, save_plots=True):
        """绘制训练曲线"""
        if not self.train_history or not self.eval_history:
            return

        # 设置风格
        plt.style.use('default')
        sns.set_palette("husl")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.experiment_name} - Training Curves', fontsize=16)

        # 训练损失
        ax1 = axes[0, 0]
        epochs = [h['epoch'] for h in self.train_history]
        ax1.plot(epochs, [h['total'] for h in self.train_history], label='Total Loss', linewidth=2)
        ax1.plot(epochs, [h['task'] for h in self.train_history], label='Task Loss', alpha=0.7)
        ax1.plot(epochs, [h['indep'] for h in self.train_history], label='Indep Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 评估指标
        ax2 = axes[0, 1]
        eval_epochs = [h['epoch'] for h in self.eval_history]
        ax2.plot(eval_epochs, [h['accuracy'] for h in self.eval_history], label='Accuracy', linewidth=2)
        ax2.plot(eval_epochs, [h['macro_f1'] for h in self.eval_history], label='Macro F1', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Evaluation Metrics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 公平性指标
        ax3 = axes[1, 0]
        ax3.plot(eval_epochs, [h.get('gini', 0) for h in self.eval_history], label='Gini', linewidth=2)
        ax3.plot(eval_epochs, [h.get('dp_aspect', 0) for h in self.eval_history], label='DP-Aspect', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Fairness Score')
        ax3.set_title('Fairness Metrics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 学习率和梯度范数
        ax4 = axes[1, 1]
        ax4.plot(epochs, [h.get('lr', 0) for h in self.train_history], label='Learning Rate', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate', color='blue')
        ax4.tick_params(axis='y', labelcolor='blue')

        ax4_twin = ax4.twinx()
        ax4_twin.plot(epochs, [h.get('grad_norm', 0) for h in self.train_history],
                     label='Grad Norm', color='red', alpha=0.7)
        ax4_twin.set_ylabel('Gradient Norm', color='red')
        ax4_twin.tick_params(axis='y', labelcolor='red')

        ax4.set_title('Learning Rate & Gradient Norm')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            plot_path = os.path.join(self.experiment_dir, "plots", "training_curves.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

        return fig

    def generate_report(self):
        """生成实验报告"""
        report = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'config': self.config,
            'best_metrics': self.best_metrics,
            'total_epochs': len(self.train_history),
            'final_metrics': self.eval_history[-1] if self.eval_history else {},
            'training_summary': {
                'avg_total_loss': np.mean([h['total'] for h in self.train_history]),
                'final_total_loss': self.train_history[-1]['total'] if self.train_history else 0,
                'loss_convergence': self._analyze_convergence()
            }
        }

        report_path = os.path.join(self.experiment_dir, "experiment_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)

        return report

    def _analyze_convergence(self):
        """分析收敛情况"""
        if len(self.train_history) < 10:
            return "insufficient_data"

        # 检查损失是否在最后10个epoch中收敛
        recent_losses = [h['total'] for h in self.train_history[-10:]]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)

        if loss_std / loss_mean < 0.01:  # 变异系数小于1%
            return "converged"
        elif loss_std / loss_mean < 0.05:  # 变异系数小于5%
            return "converging"
        else:
            return "not_converged"

    def print_summary(self):
        """打印实验总结"""
        print(f"\n{'='*60}")
        print(f"实验总结: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"时间戳: {self.timestamp}")
        print(f"总轮数: {len(self.train_history)}")

        if self.best_metrics:
            print(f"\n最佳性能 (Epoch {self.best_metrics.get('epoch', 'N/A')}):")
            print(".2f")
            print(".2f")
            if 'gini' in self.best_metrics:
                print(".4f")
                print(".4f")

        if self.train_history:
            print(f"\n最终训练损失: {self.train_history[-1]['total']:.4f}")

        print(f"实验目录: {self.experiment_dir}")
        print(f"{'='*60}\n")


class ABSAResultsManager:
    """
    ABSA实验结果管理器
    支持多个实验的对比和汇总
    """

    def __init__(self, results_dir="./experiments"):
        self.results_dir = results_dir
        self.experiments = {}

    def load_experiment(self, exp_dir):
        """加载实验结果"""
        exp_name = os.path.basename(exp_dir)

        # 加载配置
        config_file = os.path.join(exp_dir, "configs", "experiment_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            config = {}

        # 加载评估历史
        eval_log_path = os.path.join(exp_dir, "logs", "eval_log.csv")
        eval_history = []
        if os.path.exists(eval_log_path):
            with open(eval_log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    eval_history.append({k: float(v) for k, v in row.items()})

        self.experiments[exp_name] = {
            'config': config,
            'eval_history': eval_history,
            'experiment_dir': exp_dir
        }

        return exp_name

    def compare_experiments(self, exp_names, metrics=['macro_f1', 'accuracy']):
        """对比多个实验"""
        results = {}

        for exp_name in exp_names:
            if exp_name not in self.experiments:
                continue

            exp_data = self.experiments[exp_name]
            eval_history = exp_data['eval_history']

            if not eval_history:
                continue

            # 找到最佳性能
            best_result = max(eval_history, key=lambda x: x.get('macro_f1', 0))

            results[exp_name] = {
                'config': exp_data['config'],
                'best_metrics': best_result,
                'final_metrics': eval_history[-1]
            }

        return results

    def generate_comparison_report(self, exp_names, output_file="comparison_report.json"):
        """生成对比报告"""
        comparison = self.compare_experiments(exp_names)

        report = {
            'comparison_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'experiments': comparison,
            'summary': self._generate_comparison_summary(comparison)
        }

        output_path = os.path.join(self.results_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)

        return report

    def _generate_comparison_summary(self, comparison):
        """生成对比总结"""
        if not comparison:
            return {}

        # 找出最佳实验
        best_exp = max(comparison.items(),
                      key=lambda x: x[1]['best_metrics'].get('macro_f1', 0))

        summary = {
            'best_experiment': best_exp[0],
            'best_macro_f1': best_exp[1]['best_metrics'].get('macro_f1', 0),
            'num_experiments': len(comparison),
            'avg_macro_f1': np.mean([exp['best_metrics'].get('macro_f1', 0)
                                   for exp in comparison.values()])
        }

        return summary


def create_experiment_logger(experiment_name, config=None):
    """
    创建实验日志记录器的便捷函数

    Args:
        experiment_name: 实验名称
        config: 配置字典

    Returns:
        ExperimentLogger: 配置好的日志记录器
    """
    return ExperimentLogger(experiment_name, config=config)


def save_model_with_metadata(model, filepath, metadata=None):
    """
    保存模型及元数据的便捷函数

    Args:
        model: PyTorch模型
        filepath: 保存路径
        metadata: 元数据字典
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {},
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)


def load_model_with_metadata(filepath, model):
    """
    加载模型及元数据的便捷函数

    Args:
        filepath: 模型路径
        model: PyTorch模型

    Returns:
        metadata: 元数据字典
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint.get('metadata', {})
