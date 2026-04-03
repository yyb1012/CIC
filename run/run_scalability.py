"""
Step 5: 运行可扩展性实验

使用方法:
    python run_scalability.py --dataset theia --device 0
"""

import argparse
import os
import sys
import json
import time
import gc
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import psutil

DATA_COMPASS_DIR = os.path.join("results", "data_compass")


def _resolve_dataset_dir(args) -> str:
    from utils.loaddata import resolve_data_dir

    if args.data_dir:
        base = resolve_data_dir(args.data_dir)
        if os.path.exists(os.path.join(base, "metadata_cic.json")):
            return base
        candidate = os.path.join(base, args.dataset)
        if os.path.exists(os.path.join(candidate, "metadata_cic.json")):
            return candidate
        return base

    return resolve_data_dir(args.dataset)


def _build_default_model_args() -> argparse.Namespace:
    from utils.config import build_args as _build_args

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        return _build_args()
    finally:
        sys.argv = old_argv


def _load_cached_training_time(dataset: str, seed: int) -> float:
    candidates = [
        os.path.join(DATA_COMPASS_DIR, dataset, f"train_metrics_{dataset}_s{seed}.json"),
        os.path.join(DATA_COMPASS_DIR, dataset, f"train_metrics_{dataset}.json"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            value = float(payload.get("train_time_seconds", -1.0))
            if value > 0:
                return value
        except Exception:
            continue
    return -1.0


def _parse_scale_nodes(value: str) -> list:
    nodes = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        nodes.append(int(part))
    return [n for n in nodes if n > 0]


def _discover_seed_checkpoints(dataset_name: str) -> list:
    """自动发现所有 seed 的 checkpoint 文件，返回 [(seed, path), ...] 列表"""
    import re
    ckpt_dir = os.path.join("checkpoints", dataset_name)
    if not os.path.isdir(ckpt_dir):
        return []
    
    pattern = re.compile(rf"checkpoint-{re.escape(dataset_name)}_s(\d+)\.pt$")
    results = []
    for fname in os.listdir(ckpt_dir):
        m = pattern.match(fname)
        if m:
            seed = int(m.group(1))
            results.append((seed, os.path.join(ckpt_dir, fname)))
    
    results.sort(key=lambda x: x[0])
    return results


def _parse_seed_from_checkpoint(dataset_name: str, ckpt_path: str) -> int:
    import re
    fname = os.path.basename(ckpt_path)
    pattern = re.compile(rf"checkpoint-{re.escape(dataset_name)}_s(\d+)\.pt$")
    m = pattern.match(fname)
    if m:
        return int(m.group(1))
    return 0


def _format_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if np.isnan(value):
            return "n/a"
        return f"{value:.6f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _write_scalability_readable(path: str, header: dict, metrics: dict) -> None:
    lines = ["Scalability summary", ""]
    for key, value in header.items():
        lines.append(f"{key}: {_format_value(value)}")
    lines.append("")
    if metrics:
        lines.append("timing metrics:")
        timing_keys = ["train_time_seconds", "inference_avg_seconds", "inference_std_seconds",
                       "inference_nodes_per_second", "peak_gpu_mb"]
        for key in timing_keys:
            if key in metrics:
                lines.append(f"- {key}: {_format_value(metrics.get(key))}")
    else:
        lines.append("metrics: n/a")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _summarize_scalability_metrics(metrics_list: list) -> dict:
    summary = {"metrics_mean": {}, "metrics_std": {}}
    if not metrics_list:
        return summary
    
    keys = set()
    for entry in metrics_list:
        if isinstance(entry, dict):
            keys.update(entry.keys())
    
    for key in sorted(keys):
        values = []
        for entry in metrics_list:
            if not isinstance(entry, dict) or key not in entry:
                continue
            value = entry.get(key)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if not values:
            summary["metrics_mean"][key] = float("nan")
            summary["metrics_std"][key] = float("nan")
            continue
        arr = np.array(values, dtype=float)
        summary["metrics_mean"][key] = float(np.nanmean(arr))
        summary["metrics_std"][key] = float(np.nanstd(arr))
    return summary


def main():
    parser = argparse.ArgumentParser(description='运行可扩展性实验')
    parser.add_argument('--dataset', type=str, default='theia', choices=['theia', 'cadets', 'clear', 'trace'])
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--train_epochs', type=int, default=3, help='训练轮数（用于测量训练时间）')
    parser.add_argument('--inference_runs', type=int, default=3, help='推理运行次数（取平均）')
    parser.add_argument('--skip_training_timing', action='store_true', help='跳过训练时间测量')
    parser.add_argument('--force_training_timing', action='store_true', help='忽略缓存，重新测训练时间')
    parser.add_argument('--scale_nodes', type=str, default='100,500,1000,2000,5000',
                        help='可扩展性曲线的节点分桶边界，逗号分隔（使用真实日志图）')
    parser.add_argument('--scale_runs', type=int, default=3, help='推理重复次数（取平均）')
    parser.add_argument('--scale_edges_per_node', type=int, default=5, help='兼容参数（真实日志下不使用）')
    args = parser.parse_args()
    
    data_dir = _resolve_dataset_dir(args)
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.isdir(data_dir):
        os.environ.setdefault("DATA_ROOT", os.path.dirname(data_dir))
    
    print("=" * 60)
    print("可扩展性实验")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"数据目录: {data_dir}")
    print(f"设备: {'CPU' if args.device < 0 else f'cuda:{args.device}'}")
    
    # 加载数据
    from utils.loaddata import (
        load_metadata_with_cic,
        load_entity_level_dataset_with_cic,
        load_entity_level_dataset,
    )
    from model.autoencoder import build_model
    from utils.utils import create_optimizer
    
    metadata = load_metadata_with_cic(data_dir)
    n_train = int(metadata.get('n_train', 0))
    n_test = int(metadata.get('n_test', 0))
    if n_train <= 0 or n_test <= 0:
        print("[ERR] metadata中n_train或n_test无效，请确认CIC预处理是否完成。")
        return
    
    # 构建模型
    model_args = _build_default_model_args()
    model_args.dataset = args.dataset
    model_args.device = args.device
    model_args.num_hidden = 64
    model_args.num_layers = 3
    use_cic = bool(getattr(model_args, "use_cic", False)) and bool(metadata.get("has_cic_scores", False))
    add_cic_to_features = use_cic and bool(getattr(model_args, "cic_as_node_feature", False))
    model_args.n_dim = metadata['node_feature_dim'] + (4 if add_cic_to_features else 0)
    model_args.e_dim = metadata['edge_feature_dim']
    
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    
    loader = load_entity_level_dataset_with_cic if use_cic else load_entity_level_dataset

    def _append_cic_features(g):
        if add_cic_to_features and 'cic_scores' in g.ndata and 'attr' in g.ndata:
            cic = g.ndata['cic_scores'].to(device=g.ndata['attr'].device, dtype=g.ndata['attr'].dtype)
            g.ndata['attr'] = torch.cat([g.ndata['attr'], cic], dim=-1)
    
    # 发现所有种子 checkpoints
    seed_checkpoints = []
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            seed = _parse_seed_from_checkpoint(args.dataset, args.checkpoint)
            seed_checkpoints = [(seed, args.checkpoint)]
        else:
            print(f"[ERR] checkpoint not found: {args.checkpoint}")
            return
    else:
        seed_checkpoints = _discover_seed_checkpoints(args.dataset)
        if not seed_checkpoints:
            default_ckpt = os.path.join("checkpoints", args.dataset, f"checkpoint-{args.dataset}.pt")
            legacy_path = f'./checkpoints/checkpoint-{args.dataset}.pt'
            if os.path.exists(default_ckpt):
                seed_checkpoints = [(0, default_ckpt)]
            elif os.path.exists(legacy_path):
                seed_checkpoints = [(0, legacy_path)]
            else:
                seed_checkpoints = [(0, None)]
    
    print(f"[INFO] 发现 {len(seed_checkpoints)} 个种子 checkpoints")
    
    scalability_dir = os.path.join(args.output_dir, "scalability", args.dataset)
    os.makedirs(scalability_dir, exist_ok=True)
    
    all_seed_metrics = []
    all_seed_results = []
    
    for seed_value, ckpt_path in seed_checkpoints:
        print(f"\n{'='*60}")
        print(f"种子 {seed_value} 可扩展性测试")
        print(f"{'='*60}")

        seed_output_dir = os.path.join(scalability_dir, f"seed_s{seed_value}")
        os.makedirs(seed_output_dir, exist_ok=True)

        # 为每个种子重新构建模型
        model = build_model(model_args).to(device)
        optimizer = create_optimizer(model_args.optimizer, model, model_args.lr, model_args.weight_decay)

        has_ckpt = ckpt_path and os.path.exists(ckpt_path)
        if has_ckpt:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"[OK] 已加载模型: {ckpt_path}")
        else:
            print(f"[WARN] 未找到检查点 (seed={seed_value})")

        results = {
            'dataset': args.dataset,
            'seed': seed_value,
            'checkpoint': ckpt_path,
            'device': str(device),
            'timing': {},
            'memory': {},
            'scaling': {},
        }

        # ========== 测量训练时间 ==========
        print(f"\n[1/4] 测量训练时间 (epochs={args.train_epochs})...")

        train_time = None
        if args.skip_training_timing:
            print("  [SKIP] 已跳过训练时间测量")
        else:
            cached_time = _load_cached_training_time(args.dataset, seed_value)
            if cached_time > 0 and not args.force_training_timing:
                train_time = cached_time
                results['timing']['training'] = {
                    'total_seconds': train_time,
                    'epochs': None,
                    'samples': None,
                    'total_nodes': None,
                    'samples_per_second': None,
                    'nodes_per_second': None,
                    'source': 'cached',
                }
                print(f"  [OK] 使用已有训练时间记录: {train_time:.2f}s")
            else:
                model.train()
                total_samples = 0
                total_nodes = 0

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                train_start = time.perf_counter()

                for epoch in range(args.train_epochs):
                    for i in range(n_train):
                        g = loader(data_dir, 'train', i).to(device)
                        _append_cic_features(g)

                        optimizer.zero_grad()
                        loss = model(g)
                        loss.backward()
                        optimizer.step()

                        total_samples += 1
                        total_nodes += g.num_nodes()
                        del g

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                train_time = time.perf_counter() - train_start
                results['timing']['training'] = {
                    'total_seconds': train_time,
                    'epochs': args.train_epochs,
                    'samples': total_samples,
                    'total_nodes': total_nodes,
                    'samples_per_second': total_samples / train_time if train_time > 0 else 0,
                    'nodes_per_second': total_nodes / train_time if train_time > 0 else 0,
                    'source': 'benchmark',
                }
                print(f"  训练时间: {train_time:.2f}s")
                print(f"  样本/秒: {total_samples / train_time:.2f}")
                print(f"  节点/秒: {total_nodes / train_time:.2f}")

                # 恢复检查点，避免影响后续推理计时
                if has_ckpt:
                    model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # ========== 测量推理时间 ==========
        print(f"\n[2/4] 测量推理时间 (runs={args.inference_runs})...")

        model.eval()
        inference_times = []
        inference_nodes = 0
        graph_timings = []
        max_graph_index = 0
        max_graph_nodes = -1

        for run in range(args.inference_runs):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            run_start = time.perf_counter()
            run_nodes = 0

            with torch.no_grad():
                for i in range(n_test):
                    graph_start = time.perf_counter()
                    g = loader(data_dir, 'test', i).to(device)
                    _append_cic_features(g)
                    _ = model.embed(g)
                    graph_elapsed = time.perf_counter() - graph_start
                    run_nodes += g.num_nodes()
                    if run == 0:
                        graph_timings.append(
                            {
                                "n_nodes": int(g.num_nodes()),
                                "n_edges": int(g.num_edges()),
                                "time_s": float(graph_elapsed),
                            }
                        )
                        if g.num_nodes() > max_graph_nodes:
                            max_graph_nodes = g.num_nodes()
                            max_graph_index = i
                    del g

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            inference_times.append(time.perf_counter() - run_start)
            inference_nodes = run_nodes

        avg_infer_time = np.mean(inference_times)
        std_infer_time = np.std(inference_times)

        results['timing']['inference'] = {
            'avg_seconds': avg_infer_time,
            'std_seconds': std_infer_time,
            'runs': args.inference_runs,
            'total_nodes': inference_nodes,
            'nodes_per_second': inference_nodes / avg_infer_time if avg_infer_time > 0 else 0,
        }

        print(f"  推理时间: {avg_infer_time:.4f}s ± {std_infer_time:.4f}s")
        if avg_infer_time > 0:
            print(f"  节点/秒: {inference_nodes / avg_infer_time:.2f}")
        else:
            print("  节点/秒: n/a")

        # ========== 测量内存使用 ==========
        print("\n[3/4] 测量内存使用...")

        gc.collect()
        process = psutil.Process(os.getpid())
        cpu_before = process.memory_info().rss / (1024 * 1024)

        gpu_before = 0.0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gpu_before = torch.cuda.memory_allocated() / (1024 * 1024)

        # 执行推理
        g = loader(data_dir, 'test', max_graph_index).to(device)
        _append_cic_features(g)

        with torch.no_grad():
            _ = model.embed(g)

        cpu_after = process.memory_info().rss / (1024 * 1024)

        gpu_peak = 0.0
        if torch.cuda.is_available():
            gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

        results['memory'] = {
            'peak_gpu_mb': gpu_peak,
            'cpu_increase_mb': cpu_after - cpu_before,
            'graph_nodes': g.num_nodes(),
            'graph_edges': g.num_edges(),
        }

        print(f"  峰值GPU内存: {gpu_peak:.2f} MB")
        print(f"  CPU内存增量: {cpu_after - cpu_before:.2f} MB")
        print(f"  测试图: {g.num_nodes()} 节点, {g.num_edges()} 边")

        del g

        # ========== 测量多规模推理曲线 ==========
        print("\n[4/4] 测量多规模推理曲线...")
        scaling_results = []
        scale_nodes = _parse_scale_nodes(args.scale_nodes)
        if scale_nodes:
            scale_nodes = sorted(scale_nodes)
            prev = 0
            for upper in scale_nodes:
                bucket = [g for g in graph_timings if prev < g["n_nodes"] <= upper]
                if not bucket:
                    prev = upper
                    continue
                times = np.array([g["time_s"] for g in bucket], dtype=float)
                nodes = np.array([g["n_nodes"] for g in bucket], dtype=float)
                edges = np.array([g["n_edges"] for g in bucket], dtype=float)
                total_time = float(np.sum(times))
                total_nodes = float(np.sum(nodes))
                scaling_results.append(
                    {
                        "n_nodes_min": int(prev + 1),
                        "n_nodes_max": int(upper),
                        "num_graphs": int(len(bucket)),
                        "avg_nodes": float(np.mean(nodes)),
                        "avg_edges": float(np.mean(edges)),
                        "avg_time_s": float(np.mean(times)),
                        "std_time_s": float(np.std(times)),
                        "throughput_nodes_per_s": float(total_nodes / total_time) if total_time > 0 else 0.0,
                    }
                )
                prev = upper

            overflow = [g for g in graph_timings if g["n_nodes"] > (scale_nodes[-1] if scale_nodes else 0)]
            if overflow:
                times = np.array([g["time_s"] for g in overflow], dtype=float)
                nodes = np.array([g["n_nodes"] for g in overflow], dtype=float)
                edges = np.array([g["n_edges"] for g in overflow], dtype=float)
                total_time = float(np.sum(times))
                total_nodes = float(np.sum(nodes))
                scaling_results.append(
                    {
                        "n_nodes_min": int(scale_nodes[-1] + 1),
                        "n_nodes_max": None,
                        "num_graphs": int(len(overflow)),
                        "avg_nodes": float(np.mean(nodes)),
                        "avg_edges": float(np.mean(edges)),
                        "avg_time_s": float(np.mean(times)),
                        "std_time_s": float(np.std(times)),
                        "throughput_nodes_per_s": float(total_nodes / total_time) if total_time > 0 else 0.0,
                    }
                )
        else:
            print("  [SKIP] no scale_nodes configured")

        results['scaling']['inference'] = scaling_results

        # 保存单个种子结果
        output_path = os.path.join(seed_output_dir, f'scalability_{args.dataset}_s{seed_value}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] 结果已保存: {output_path}")

        # 提取关键指标用于汇总
        seed_metrics = {
            "seed": seed_value,
            "train_time_seconds": train_time,
            "inference_avg_seconds": avg_infer_time,
            "inference_std_seconds": std_infer_time,
            "inference_nodes_per_second": inference_nodes / avg_infer_time if avg_infer_time > 0 else 0,
            "peak_gpu_mb": gpu_peak,
        }
        all_seed_metrics.append(seed_metrics)
        all_seed_results.append(results)

        # 写入可读摘要
        readable_path = os.path.join(seed_output_dir, f'scalability_summary_{args.dataset}_s{seed_value}_readable.txt')
        _write_scalability_readable(
            readable_path,
            {
                "dataset": args.dataset,
                "seed": seed_value,
                "checkpoint": ckpt_path,
                "device": str(device),
            },
            seed_metrics,
        )

        # 打印种子摘要
        print(f"\n[种子 {seed_value} 结果摘要]")
        train_time_display = f"{train_time:.2f}s" if train_time is not None else "n/a"
        print(f"  训练时间: {train_time_display}")
        print(f"  推理时间: {avg_infer_time:.4f}s ± {std_infer_time:.4f}s")
        print(f"  峰值GPU内存: {gpu_peak:.2f} MB")
        if avg_infer_time > 0:
            print(f"  吞吐量: {inference_nodes/avg_infer_time:.0f} 节点/秒")
        else:
            print("  吞吐量: n/a")
    
    # ========== 汇总所有种子结果 ==========
    if len(all_seed_metrics) > 1:
        print(f"\n{'='*60}")
        print(f"汇总 {len(all_seed_metrics)} 个种子的可扩展性指标")
        print(f"{'='*60}")
        
        summary = _summarize_scalability_metrics(all_seed_metrics)
        summary_data = {
            "dataset": args.dataset,
            "num_seeds": len(all_seed_metrics),
            "seeds": [m["seed"] for m in all_seed_metrics],
            "device": str(device),
            **summary,
        }
        
        summary_path = os.path.join(scalability_dir, f'scalability_summary_{args.dataset}_all_seeds.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"[OK] 汇总已保存: {summary_path}")
        
        # 写入汇总可读摘要
        readable_summary_path = os.path.join(scalability_dir, f'scalability_summary_{args.dataset}_all_seeds_readable.txt')
        lines = ["Scalability summary (all seeds)", ""]
        lines.append(f"dataset: {args.dataset}")
        lines.append(f"num_seeds: {len(all_seed_metrics)}")
        lines.append(f"seeds: {[m['seed'] for m in all_seed_metrics]}")
        lines.append(f"device: {str(device)}")
        lines.append("")
        lines.append("metrics (mean ± std):")
        means = summary.get("metrics_mean", {})
        stds = summary.get("metrics_std", {})
        for key in ["train_time_seconds", "inference_avg_seconds", "inference_nodes_per_second", "peak_gpu_mb"]:
            if key in means:
                lines.append(f"- {key}: {_format_value(means.get(key))} ± {_format_value(stds.get(key))}")
        with open(readable_summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        # 打印汇总表格
        print("\n可扩展性实验汇总结果 (mean ± std)")
        print("-" * 60)
        for key in ["train_time_seconds", "inference_avg_seconds", "inference_nodes_per_second", "peak_gpu_mb"]:
            if key in means:
                print(f"  {key}: {means[key]:.4f} ± {stds[key]:.4f}")
    else:
        # 单种子情况也写入汇总文件
        if all_seed_results:
            single = all_seed_results[0]
            summary_path = os.path.join(scalability_dir, f'scalability_summary_{args.dataset}_all_seeds.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "dataset": args.dataset,
                    "num_seeds": 1,
                    "seeds": [all_seed_metrics[0]["seed"]] if all_seed_metrics else [0],
                    "device": str(device),
                    **single,
                }, f, indent=2, ensure_ascii=False)
            print(f"[OK] 汇总已保存: {summary_path}")
    
    print("\n" + "=" * 60)
    print("可扩展性实验完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
