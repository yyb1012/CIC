"""
独立可视化脚本 - 生成论文图表

从 run_experiments.py 抽离，可单独运行生成：
- ROC曲线 / PR曲线
- CIC分数分布箱线图
- 消融实验对比柱状图
- 鲁棒性对比折线图

使用方法:
    python run/run_visualization.py --data_dir /hy-tmp/code/data --dataset theia
    python run/run_visualization.py --data_dir /hy-tmp/code/data --dataset theia --results_dir ./results
"""

import argparse
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.visualization import generate_all_plots


def _resolve_results_dir(args) -> str:
    """解析结果目录"""
    if args.results_dir:
        base = os.path.abspath(args.results_dir)
        # 检查是否已包含 dataset 子目录
        if os.path.basename(base) == args.dataset:
            return base
        candidate = os.path.join(base, args.dataset)
        if os.path.exists(candidate):
            return candidate
        return base
    # 默认目录
    return os.path.join("results", args.dataset)


def run_visualization(args) -> dict:
    """运行可视化生成"""
    results_dir = _resolve_results_dir(args)
    
    if not os.path.exists(results_dir):
        print(f"[ERR] 结果目录不存在: {results_dir}")
        return {}
    
    print(f"[INFO] 结果目录: {results_dir}")
    
    # 尝试加载消融实验结果
    ablation_data = {}
    ablation_paths = [
        os.path.join(results_dir, "ablation", "ablation_results.json"),
        os.path.join(results_dir, "ablation_results.json"),
        os.path.join(results_dir, f"ablation_summary_{args.dataset}_all_seeds.json"),
    ]
    
    for path in ablation_paths:
        if os.path.exists(path):
            print(f"[OK] 加载消融结果: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                ablation_data = json.load(f)
            break
    
    if not ablation_data:
        print("[WARN] 未找到消融实验结果，部分图表可能为空")
    
    # 尝试加载检测结果
    detection_data = {}
    detection_paths = [
        os.path.join(results_dir, "detection_results.json"),
        os.path.join(results_dir, "eval_results.json"),
    ]
    
    for path in detection_paths:
        if os.path.exists(path):
            print(f"[OK] 加载检测结果: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                detection_data = json.load(f)
            break
    
    # 尝试加载鲁棒性结果
    robustness_data = {}
    robustness_paths = [
        os.path.join(results_dir, "robustness", f"robustness_summary_{args.dataset}_all_seeds.json"),
        os.path.join(results_dir, "robustness_results.json"),
    ]
    
    for path in robustness_paths:
        if os.path.exists(path):
            print(f"[OK] 加载鲁棒性结果: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                robustness_data = json.load(f)
            break
    
    # 合并所有数据
    all_data = {
        "ablation": ablation_data,
        "detection": detection_data,
        "robustness": robustness_data,
    }
    
    # 生成图表
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[INFO] 生成图表到: {output_dir}")
    plots = generate_all_plots(all_data, output_dir=output_dir)
    
    print(f"\n[OK] 生成了 {len(plots)} 个图表")
    for name, path in plots.items():
        if path and os.path.exists(path):
            print(f"  - {name}: {path}")
    
    return plots


def main():
    parser = argparse.ArgumentParser(description="生成论文可视化图表")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据集目录 (可选，用于自动定位结果)")
    parser.add_argument("--dataset", type=str, default="theia",
                        choices=["theia", "cadets", "clear", "trace"],
                        help="数据集名称")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="结果目录 (包含 ablation_results.json 等)")
    
    args = parser.parse_args()
    
    run_visualization(args)


if __name__ == "__main__":
    main()
