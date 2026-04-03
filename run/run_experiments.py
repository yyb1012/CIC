"""
Run detection, robustness, and visualization experiments (no ablation).

Use run/run_ablation.py for ablation experiments to avoid overlap.
"""

import argparse
import os
import sys
import json
import numpy as np
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch

from experiments.evaluation import (
    EntityLevelEvaluator,
    save_evaluation_results,
    load_cic_scores_and_labels,
)
from experiments.robustness import (
    run_full_robustness_study,
    SemanticPerturbation,
    PerturbationConfig,
)
from experiments.visualization import generate_all_plots


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


def run_detection_experiment(pred_scores: np.ndarray,
                             labels: np.ndarray,
                             output_dir: str) -> dict:
    evaluator = EntityLevelEvaluator()

    result = evaluator.evaluate(pred_scores, labels)
    best_thr, best_result = evaluator.find_optimal_threshold(pred_scores, labels)

    results = {
        "basic": result.to_dict(),
        "optimal_threshold": float(best_thr),
        "optimal_result": best_result.to_dict(),
    }
    save_evaluation_results(results, os.path.join(output_dir, "detection_results.json"))
    return results


def _compute_fused_scores(model,
                          graph,
                          device,
                          *,
                          node_feature_dim: int,
                          add_cic_to_features: bool,
                          use_cic_scores: bool,
                          node_contrast,
                          anomaly_scorer,
                          anomaly_threshold: float) -> np.ndarray:
    """
    计算模型在给定图上的完整融合分数 (CIC + 对比 + 重构)
    """
    model.eval()
    g = graph.to(device)
    
    if add_cic_to_features:
        if 'attr' not in g.ndata:
            raise ValueError("Graph missing ndata['attr']")
        expected_dim = int(node_feature_dim) + 4
        if g.ndata['attr'].shape[1] != expected_dim:
            if 'cic_scores' in g.ndata:
                cic = g.ndata['cic_scores'].to(device=g.ndata['attr'].device, dtype=g.ndata['attr'].dtype)
            else:
                cic = torch.zeros((g.num_nodes(), 4), device=g.ndata['attr'].device, dtype=g.ndata['attr'].dtype)
            if not use_cic_scores:
                cic = torch.zeros_like(cic)
            g.ndata['attr'] = torch.cat([g.ndata['attr'], cic], dim=-1)

    if 'cic_scores' in g.ndata:
        cic_scores = g.ndata['cic_scores'].to(device=g.device, dtype=torch.float32)
    else:
        cic_scores = torch.zeros((g.num_nodes(), 4), device=g.device, dtype=torch.float32)

    if not use_cic_scores:
        cic_for_scoring = torch.zeros_like(cic_scores)
    else:
        cic_for_scoring = cic_scores
    
    with torch.no_grad():
        emb = model.embed(g)
        if isinstance(emb, tuple):
            emb = emb[0]
        if hasattr(model, "node_reconstruction_error"):
            recon_error = model.node_reconstruction_error(g)
        else:
            recon_error = torch.zeros(g.num_nodes(), device=g.device)
        contrast_score = node_contrast.anomaly_score(
            emb, cic_for_scoring, threshold=float(anomaly_threshold)
        )
        fused = anomaly_scorer.compute_anomaly_score(
            cic_for_scoring, contrastive_score=contrast_score, recon_error=recon_error
        )
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return fused.detach().cpu().numpy()


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
    
    # 按 seed 排序
    results.sort(key=lambda x: x[0])
    return results


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


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


def _write_robustness_readable(path: str, header: Dict[str, Any], summary: Dict[str, Any]) -> None:
    lines = ["Robustness summary", ""]
    for key, value in header.items():
        lines.append(f"{key}: {_format_value(value)}")
    lines.append("")
    if summary:
        lines.append("summary metrics (perturbed - original; higher robustness is better):")
        keys = [
            "avg_baseline_delta_f1",
            "avg_cic_delta_f1",
            "avg_improvement",
            "avg_baseline_robustness",
            "avg_cic_robustness",
            "avg_robustness_gain",
        ]
        for key in keys:
            if key in summary:
                lines.append(f"- {key}: {_format_value(summary.get(key))}")
    else:
        lines.append("summary metrics: n/a")
    lines.append("")
    lines.append("notes:")
    lines.append("- delta_*: perturbed - original (negative means worse)")
    lines.append("- robustness_score: 0-1, higher is better")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _run_single_seed_robustness(
    args,
    data_dir: str,
    ckpt_path: str,
    seed_value: int,
    metadata: dict,
    device: torch.device,
) -> Dict[str, Any]:
    """
    对单个种子运行鲁棒性实验
    """
    from utils.loaddata import (
        load_entity_level_dataset_with_cic,
        load_entity_level_dataset,
    )
    from model.autoencoder import build_model
    from model.contrastive import NodeLevelContrastive
    from model.fusion import AnomalyScorer
    import pickle as pkl
    
    n_test = int(metadata.get('n_test', 0))
    
    # 构建模型
    model_args = _build_default_model_args()
    model_args.dataset = args.dataset
    model_args.device = getattr(args, 'device', 0)
    model_args.num_hidden = 64
    model_args.num_layers = 3
    use_cic = bool(getattr(model_args, "use_cic", False)) and bool(metadata.get("has_cic_scores", False))
    add_cic_to_features = use_cic and bool(getattr(model_args, "cic_as_node_feature", False))
    node_feature_dim = int(metadata['node_feature_dim'])
    model_args.n_dim = node_feature_dim + (4 if add_cic_to_features else 0)
    model_args.e_dim = metadata['edge_feature_dim']
    
    model = build_model(model_args).to(device)
    node_contrast = NodeLevelContrastive(hidden_dim=int(getattr(model, "output_hidden_dim", 64))).to(device)
    anomaly_scorer = AnomalyScorer(n_sources=3).to(device)
    
    # 加载检查点
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"[WARN] 未找到检查点 {ckpt_path}")
        return {}
    
    model.eval()
    
    loader = load_entity_level_dataset_with_cic if use_cic else load_entity_level_dataset
    perturbation = SemanticPerturbation(seed=42 + seed_value)  # 不同种子用不同扰动
    
    # 加载恶意标签
    malicious_path = os.path.join(data_dir, 'malicious.pkl')
    malicious_set = set()
    if os.path.exists(malicious_path):
        with open(malicious_path, 'rb') as f:
            malicious = pkl.load(f)
        if isinstance(malicious, tuple) and len(malicious) == 2:
            malicious_set = set(int(x) for x in malicious[0])
        elif isinstance(malicious, list):
            malicious_set = set(int(x) for x in malicious)
    
    # 收集原始分数和扰动分数
    baseline_scores = {
        "original": [],
        "perturbed_path": [],
        "perturbed_cmd": [],
        "perturbed_alias": [],
        "perturbed_mixed": [],
    }
    cic_scores_dict = {
        "original": [],
        "perturbed_path": [],
        "perturbed_cmd": [],
        "perturbed_alias": [],
        "perturbed_mixed": [],
    }
    all_labels = []
    total_nodes = 0
    skip_benign = 0
    
    for i in range(n_test):
        g_original = loader(data_dir, 'test', i)
        if i != n_test - 1:
            skip_benign += g_original.num_nodes()
        
        # 原始分数 (baseline = 不用CIC特征, CIC = 用CIC特征)
        scores_baseline_orig = _compute_fused_scores(
            model,
            g_original,
            device,
            node_feature_dim=node_feature_dim,
            add_cic_to_features=add_cic_to_features,
            use_cic_scores=False,
            node_contrast=node_contrast,
            anomaly_scorer=anomaly_scorer,
            anomaly_threshold=float(getattr(args, "cic_anomaly_threshold", 0.5)),
        )
        scores_cic_orig = _compute_fused_scores(
            model,
            g_original,
            device,
            node_feature_dim=node_feature_dim,
            add_cic_to_features=add_cic_to_features,
            use_cic_scores=add_cic_to_features,
            node_contrast=node_contrast,
            anomaly_scorer=anomaly_scorer,
            anomaly_threshold=float(getattr(args, "cic_anomaly_threshold", 0.5)),
        )
        
        baseline_scores["original"].append(scores_baseline_orig)
        cic_scores_dict["original"].append(scores_cic_orig)
        
        # 构建标签
        labels = np.zeros(g_original.num_nodes())
        offset = total_nodes
        for nid in range(g_original.num_nodes()):
            if (offset + nid) in malicious_set:
                labels[nid] = 1
        all_labels.append(labels)
        total_nodes += g_original.num_nodes()
        
        # 扰动实验
        perturbation_types = ['path', 'cmd', 'alias', 'mixed']
        for ptype in perturbation_types:
            g_perturbed = g_original.clone()
            
            if 'attr' in g_perturbed.ndata:
                perturbation_rate = 0.3
                perturbation_strength = 0.5
                if ptype == 'mixed':
                    perturbation_rate = 0.5
                    perturbation_strength = 0.7
                perturbed_attr = perturbation.perturb_node_features(
                    g_perturbed.ndata['attr'],
                    perturbation_rate=perturbation_rate,
                    perturbation_strength=perturbation_strength,
                )
                g_perturbed.ndata['attr'] = perturbed_attr

            scores_baseline_pert = _compute_fused_scores(
                model,
                g_perturbed,
                device,
                node_feature_dim=node_feature_dim,
                add_cic_to_features=add_cic_to_features,
                use_cic_scores=False,
                node_contrast=node_contrast,
                anomaly_scorer=anomaly_scorer,
                anomaly_threshold=float(getattr(args, "cic_anomaly_threshold", 0.5)),
            )
            scores_cic_pert = _compute_fused_scores(
                model,
                g_perturbed,
                device,
                node_feature_dim=node_feature_dim,
                add_cic_to_features=add_cic_to_features,
                use_cic_scores=add_cic_to_features,
                node_contrast=node_contrast,
                anomaly_scorer=anomaly_scorer,
                anomaly_threshold=float(getattr(args, "cic_anomaly_threshold", 0.5)),
            )
            
            baseline_scores[f"perturbed_{ptype}"].append(scores_baseline_pert)
            cic_scores_dict[f"perturbed_{ptype}"].append(scores_cic_pert)
    
    # 合并所有分数
    for key in baseline_scores:
        baseline_scores[key] = np.concatenate(baseline_scores[key], axis=0)
        cic_scores_dict[key] = np.concatenate(cic_scores_dict[key], axis=0)
    
    labels = np.concatenate(all_labels, axis=0)
    keep_mask = (np.arange(total_nodes) >= skip_benign) | (labels == 1)
    labels = labels[keep_mask]
    for key in baseline_scores:
        baseline_scores[key] = baseline_scores[key][keep_mask]
        cic_scores_dict[key] = cic_scores_dict[key][keep_mask]
    
    return {
        "labels": labels,
        "baseline_scores": baseline_scores,
        "cic_scores": cic_scores_dict,
        "n_entities": len(labels),
        "n_malicious": int(labels.sum()),
    }


def run_robustness_experiment(args, data_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    运行真实的鲁棒性实验：对图特征做扰动后重新推理（支持多种子）
    """
    from utils.loaddata import load_metadata_with_cic
    from tqdm import tqdm
    
    print("\n" + "=" * 60)
    print("鲁棒性实验 (真实图扰动 - 多种子)")
    print("=" * 60)
    
    # 加载元数据
    metadata = load_metadata_with_cic(data_dir)
    n_test = int(metadata.get('n_test', 0))
    
    if n_test <= 0:
        print("[ERR] n_test 无效")
        return {}
    
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    
    # 发现所有种子 checkpoints
    seed_checkpoints = []
    if getattr(args, "checkpoint", None):
        if os.path.exists(args.checkpoint):
            seed_checkpoints = [(0, args.checkpoint)]
        else:
            print(f"[ERR] checkpoint not found: {args.checkpoint}")
            return {}
    else:
        seed_checkpoints = _discover_seed_checkpoints(args.dataset)
    if not seed_checkpoints:
        # 尝试用单个默认 checkpoint
        default_ckpt = os.path.join("checkpoints", args.dataset, f"checkpoint-{args.dataset}.pt")
        if os.path.exists(default_ckpt):
            seed_checkpoints = [(0, default_ckpt)]
        else:
            print(f"[ERR] 未找到 {args.dataset} 的任何 checkpoint")
            return {}
    
    print(f"[INFO] 发现 {len(seed_checkpoints)} 个种子 checkpoints")
    
    # 对每个种子运行鲁棒性实验
    all_seed_results = []
    robustness_dir = os.path.join(output_dir, "robustness", args.dataset)
    os.makedirs(robustness_dir, exist_ok=True)
    
    seed_iter = tqdm(seed_checkpoints, desc="Seeds") if len(seed_checkpoints) > 1 else seed_checkpoints
    
    for seed_value, ckpt_path in seed_iter:
        print(f"\n[SEED {seed_value}] 加载 {ckpt_path}")
        
        seed_result = _run_single_seed_robustness(
            args, data_dir, ckpt_path, seed_value, metadata, device
        )
        
        if not seed_result:
            continue
        
        # 对该种子运行鲁棒性对比
        single_result = run_full_robustness_study(
            labels=seed_result["labels"],
            baseline_scores=seed_result["baseline_scores"],
            cic_scores=seed_result["cic_scores"],
            perturbation_rates=[0.3],
            perturbation_strengths=[0.5],
            output_dir=os.path.join(robustness_dir, f"seed_s{seed_value}"),
        )
        
        # 保存单个种子结果
        single_result["seed"] = seed_value
        single_result["checkpoint"] = ckpt_path
        single_result["n_entities"] = seed_result["n_entities"]
        single_result["n_malicious"] = seed_result["n_malicious"]
        
        save_path = os.path.join(robustness_dir, f"robustness_{args.dataset}_s{seed_value}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(_to_serializable(single_result), f, indent=2, ensure_ascii=False)
        print(f"[OK] 已保存: {save_path}")

        readable_path = os.path.join(
            robustness_dir,
            f"seed_s{seed_value}",
            f"robustness_summary_{args.dataset}_s{seed_value}_readable.txt",
        )
        _write_robustness_readable(
            readable_path,
            {
                "dataset": args.dataset,
                "seed": seed_value,
                "checkpoint": ckpt_path,
                "n_entities": single_result.get("n_entities"),
                "n_malicious": single_result.get("n_malicious"),
            },
            single_result.get("summary", {}),
        )
        
        all_seed_results.append(single_result)
    
    # 汇总所有种子结果 (mean ± std)
    if len(all_seed_results) > 1:
        print(f"\n[汇总] 计算 {len(all_seed_results)} 个种子的 mean ± std...")
        
        # 收集关键指标（来自 run_full_robustness_study 的 summary）
        metric_keys = [
            "avg_baseline_delta_f1",
            "avg_cic_delta_f1",
            "avg_improvement",
            "avg_baseline_robustness",
            "avg_cic_robustness",
            "avg_robustness_gain",
        ]
        summary = {
            "dataset": args.dataset,
            "num_seeds": len(all_seed_results),
            "seeds": [r["seed"] for r in all_seed_results],
        }
        
        for key in metric_keys:
            values = []
            for r in all_seed_results:
                summary_block = r.get("summary", {}) if isinstance(r, dict) else {}
                if key in summary_block:
                    values.append(float(summary_block[key]))
            
            if values:
                summary[key] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
        
        # 保存汇总结果
        summary_path = os.path.join(robustness_dir, f"robustness_summary_{args.dataset}_all_seeds.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[OK] 已保存汇总: {summary_path}")

        readable_path = os.path.join(
            robustness_dir,
            f"robustness_summary_{args.dataset}_all_seeds_readable.txt",
        )
        _write_robustness_readable(
            readable_path,
            {
                "dataset": args.dataset,
                "num_seeds": len(all_seed_results),
                "seeds": [r["seed"] for r in all_seed_results],
            },
            summary,
        )
        
        return {"summary": summary, "seed_results": all_seed_results}
    
    if all_seed_results:
        single = all_seed_results[0]
        readable_path = os.path.join(
            robustness_dir,
            f"robustness_summary_{args.dataset}_all_seeds_readable.txt",
        )
        _write_robustness_readable(
            readable_path,
            {
                "dataset": args.dataset,
                "num_seeds": 1,
                "seeds": [single.get("seed")],
            },
            single.get("summary", {}),
        )
        return single
    return {}


def run_experiments(args) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)

    data_dir = _resolve_dataset_dir(args)
    if os.path.isdir(data_dir):
        os.environ.setdefault("DATA_ROOT", os.path.dirname(data_dir))

    all_results: Dict[str, Any] = {}

    # 检测实验
    if not args.skip_detection:
        print("\n[Detection] 加载数据并评估...")
        cic_scores_4d, labels, pred_scores = load_cic_scores_and_labels(data_dir)
        detection_results = run_detection_experiment(pred_scores, labels, args.output_dir)
        all_results["detection"] = detection_results

    # 鲁棒性实验 (真实图扰动)
    if not args.skip_robustness and args.run_robustness:
        robustness_results = run_robustness_experiment(args, data_dir, args.output_dir)
        all_results["robustness"] = robustness_results

    # 可视化
    if not args.skip_visualization:
        # 如果有消融结果，尝试加载
        ablation_path = os.path.join(args.output_dir, "ablation", "ablation_results.json")
        ablation_data = {}
        if os.path.exists(ablation_path):
            with open(ablation_path, 'r', encoding='utf-8') as f:
                ablation_data = json.load(f)
        
        plots = generate_all_plots(ablation_data, output_dir=os.path.join(args.output_dir, "plots"))
        all_results["plots"] = plots

    save_evaluation_results(all_results, os.path.join(args.output_dir, "all_results.json"))
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run detection/robustness/visualization experiments")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset dir or dataset root")
    parser.add_argument("--dataset", type=str, default="theia", choices=["theia", "cadets", "clear", "trace"])
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--device", type=int, default=0, help="GPU device")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--cic_anomaly_threshold", type=float, default=0.5,
                        help="Contrastive normal/anomaly split threshold")

    parser.add_argument("--skip_detection", action="store_true", help="Skip detection evaluation")
    parser.add_argument("--skip_robustness", action="store_true", help="Skip robustness evaluation")
    parser.add_argument("--run_robustness", action="store_true", help="Run robustness evaluation")
    parser.add_argument("--skip_visualization", action="store_true", help="Skip visualization output")

    args = parser.parse_args()

    run_experiments(args)


if __name__ == "__main__":
    main()
