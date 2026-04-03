"""
Step 4: 运行消融实验

使用方法:
    python run_ablation.py --dataset theia --device 0
"""

import argparse
import os
import sys
import json
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


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


def _load_malicious_indices(data_dir: str) -> list:
    import pickle as pkl

    malicious_path = os.path.join(data_dir, 'malicious.pkl')
    if not os.path.exists(malicious_path):
        return []

    with open(malicious_path, 'rb') as f:
        malicious = pkl.load(f)

    if isinstance(malicious, tuple) and len(malicious) == 2 and isinstance(malicious[0], (list, tuple)):
        return [int(x) for x in malicious[0]]

    if isinstance(malicious, list):
        return [int(x) for x in malicious]

    if isinstance(malicious, dict) and 'nodes' in malicious:
        return [int(x) for x in malicious.get('nodes', [])]

    return []


def _isotonic_regression(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y
    values = y.astype(float).tolist()
    blocks = []
    for idx, val in enumerate(values):
        blocks.append([val, 1.0, idx, idx])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            v1, w1, s1, e1 = blocks[-2]
            v2, w2, s2, e2 = blocks[-1]
            w = w1 + w2
            v = (v1 * w1 + v2 * w2) / w
            blocks[-2] = [v, w, s1, e2]
            blocks.pop()
    out = np.empty_like(y, dtype=float)
    for v, _, s, e in blocks:
        out[s:e + 1] = v
    return out


def _apply_ranking_adjustment(scores: np.ndarray, cic_scores_4d: np.ndarray) -> np.ndarray:
    from experiments.evaluation import risk_amplification_total

    cic_total = risk_amplification_total(cic_scores_4d)
    order = np.argsort(cic_total)
    adjusted = _isotonic_regression(scores[order])
    out = np.empty_like(scores, dtype=float)
    out[order] = adjusted
    return np.clip(out, 0.0, 1.0)


def _discover_seed_checkpoints(dataset_name: str) -> list:
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


def _extract_ablation_metrics(results: dict) -> dict:
    metrics = {}
    if not isinstance(results, dict):
        return metrics

    cic = results.get("cic_distribution", {})
    if isinstance(cic, dict):
        metrics["cic_only_auroc"] = cic.get("cic_only_auroc")
        metrics["cic_only_prauc"] = cic.get("cic_only_prauc")
        separation = cic.get("separation", {})
        if isinstance(separation, dict):
            metrics["cic_separation_mean_diff"] = separation.get("mean_diff")
        stats = cic.get("statistics", {})
        if isinstance(stats, dict):
            for label in ("benign", "malicious"):
                part = stats.get(label, {})
                if isinstance(part, dict):
                    metrics[f"cic_{label}_mean"] = part.get("mean")
                    metrics[f"cic_{label}_std"] = part.get("std")

    def _collect_baseline(prefix, block):
        baseline = block.get("baseline_result", {}) if isinstance(block, dict) else {}
        if isinstance(baseline, dict):
            for key in ("f1", "roc_auc", "pr_auc", "precision", "recall", "fpr"):
                if key in baseline:
                    metrics[f"{prefix}.baseline_{key}"] = baseline.get(key)

    def _collect_delta(prefix, block):
        delta = block.get("delta_metrics", {}) if isinstance(block, dict) else {}
        if isinstance(delta, dict):
            for name, values in delta.items():
                if not isinstance(values, dict):
                    continue
                for key in (
                    "delta_f1",
                    "delta_pr_auc",
                    "delta_roc_auc",
                    "delta_precision",
                    "delta_recall",
                    "delta_fpr",
                ):
                    if key in values:
                        metrics[f"{prefix}.{name}.{key}"] = values.get(key)

    inv = results.get("invariant_ablation", {})
    if isinstance(inv, dict):
        _collect_baseline("invariant", inv)
        _collect_delta("invariant", inv)

    mod = results.get("module_ablation", {})
    if isinstance(mod, dict):
        _collect_baseline("module", mod)
        _collect_delta("module", mod)

    fusion = results.get("fusion_ablation", {})
    if isinstance(fusion, dict):
        _collect_baseline("fusion", fusion)
        _collect_delta("fusion", fusion)

    return metrics


def _summarize_metrics(metrics_list: list) -> dict:
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
            values.append(float(value))
        if not values:
            summary["metrics_mean"][key] = float("nan")
            summary["metrics_std"][key] = float("nan")
            continue
        arr = np.array(values, dtype=float)
        summary["metrics_mean"][key] = float(np.nanmean(arr))
        summary["metrics_std"][key] = float(np.nanstd(arr))
    return summary


def _format_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if np.isnan(value):
            return "n/a"
        return f"{value:.6f}"
    return str(value)


def _write_ablation_readable(path: str, header: dict, metrics: dict) -> None:
    lines = ["Ablation summary", ""]
    for key, value in header.items():
        lines.append(f"{key}: {_format_value(value)}")
    lines.append("")
    if metrics:
        lines.append("metrics:")
        for key in sorted(metrics.keys()):
            lines.append(f"- {key}: {_format_value(metrics.get(key))}")
    else:
        lines.append("metrics: n/a")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_ablation_summary_readable(path: str, header: dict, summary: dict) -> None:
    lines = ["Ablation summary (all seeds)", ""]
    for key, value in header.items():
        lines.append(f"{key}: {_format_value(value)}")
    lines.append("")
    means = summary.get("metrics_mean", {}) if isinstance(summary, dict) else {}
    stds = summary.get("metrics_std", {}) if isinstance(summary, dict) else {}
    if means:
        lines.append("metrics (mean ± std):")
        for key in sorted(means.keys()):
            mean_val = means.get(key)
            std_val = stds.get(key)
            lines.append(f"- {key}: {_format_value(mean_val)} ± {_format_value(std_val)}")
    else:
        lines.append("metrics: n/a")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description='运行消融实验')
    parser.add_argument('--dataset', type=str, default='theia', choices=['theia', 'cadets', 'clear', 'trace'])
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./results/ablation')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--cic_anomaly_threshold', type=float, default=0.5,
                        help='对比学习中划分正常/异常的阈值')
    parser.add_argument('--fusion_top_k', type=int, default=10, help='融合器消融Top-k大小')
    args = parser.parse_args()
    
    data_dir = _resolve_dataset_dir(args)
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.isdir(data_dir):
        os.environ.setdefault("DATA_ROOT", os.path.dirname(data_dir))
    
    print("=" * 60)
    print("消融实验")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {args.output_dir}")
    
    # 加载CIC分数与标签（按 test_cic*.pkl 节点顺序对齐）
    print("\n[1/3] 加载CIC分数与标签...")
    from experiments.evaluation import load_cic_scores_and_labels

    cic_scores_4d, labels, pred_scores_cic = load_cic_scores_and_labels(data_dir)
    print(f"[OK] 共 {len(labels)} 个实体, {int(labels.sum())} 个恶意")

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

    from experiments.ablation import run_full_ablation_study
    from experiments.evaluation import save_evaluation_results
    from utils.loaddata import (
        load_metadata_with_cic,
        load_entity_level_dataset_with_cic,
        load_entity_level_dataset,
    )
    from model.autoencoder import build_model
    from model.contrastive import NodeLevelContrastive
    from model.fusion import AnomalyScorer

    seed_summaries = []
    for seed_value, ckpt_path in seed_checkpoints:
        dataset_output_dir = os.path.join(args.output_dir, args.dataset)
        seed_output_dir = os.path.join(dataset_output_dir, f"seed_s{seed_value}")
        os.makedirs(seed_output_dir, exist_ok=True)

        pred_scores_full = pred_scores_cic
        pred_scores_variants = None
        embeddings_kept = None
        labels_kept = labels
        cic_scores_kept = cic_scores_4d

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"\n[2/3] 计算模块消融所需分数 (seed={seed_value})...")

            metadata = load_metadata_with_cic(data_dir)
            n_test = int(metadata.get('n_test', 0))

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
            model = build_model(model_args).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            loader = load_entity_level_dataset_with_cic if use_cic else load_entity_level_dataset
            node_contrast = NodeLevelContrastive(hidden_dim=int(getattr(model, "output_hidden_dim", 64))).to(device)
            anomaly_scorer = AnomalyScorer(n_sources=3).to(device)

            def _append_cic_features(g):
                if add_cic_to_features and 'cic_scores' in g.ndata and 'attr' in g.ndata:
                    cic = g.ndata['cic_scores'].to(device=g.ndata['attr'].device, dtype=g.ndata['attr'].dtype)
                    g.ndata['attr'] = torch.cat([g.ndata['attr'], cic], dim=-1)

            full_list = []
            no_mask_list = []
            no_contrast_list = []
            emb_list = []
            cic_list = []
            total_nodes = 0
            skip_benign = 0

            with torch.no_grad():
                for i in range(n_test):
                    g = loader(data_dir, 'test', i).to(device)
                    if i != n_test - 1:
                        skip_benign += g.num_nodes()
                    _append_cic_features(g)

                    cic_scores = g.ndata.get('cic_scores')
                    if cic_scores is None:
                        cic_scores = torch.zeros((g.num_nodes(), 4), device=device)

                    emb = model.embed(g)
                    if isinstance(emb, tuple):
                        emb = emb[0]

                    contrast_score = node_contrast.anomaly_score(
                        emb, cic_scores, threshold=float(args.cic_anomaly_threshold)
                    )
                    recon_error = model.node_reconstruction_error(g)

                    full_raw = anomaly_scorer.compute_anomaly_score(
                        cic_scores, contrastive_score=contrast_score, recon_error=recon_error
                    )
                    no_mask_raw = anomaly_scorer.compute_anomaly_score(
                        cic_scores, contrastive_score=contrast_score, recon_error=None
                    )
                    no_contrast_raw = anomaly_scorer.compute_anomaly_score(
                        cic_scores, contrastive_score=None, recon_error=recon_error
                    )

                    full_list.append(full_raw.detach().cpu().numpy())
                    no_mask_list.append(no_mask_raw.detach().cpu().numpy())
                    no_contrast_list.append(no_contrast_raw.detach().cpu().numpy())
                    emb_list.append(emb.detach().cpu().numpy())
                    cic_list.append(cic_scores.detach().cpu().numpy())

                    total_nodes += g.num_nodes()
                    del g

            full_raw = np.concatenate(full_list, axis=0) if full_list else np.empty((0,), dtype=np.float32)
            no_mask_raw = np.concatenate(no_mask_list, axis=0) if no_mask_list else np.empty((0,), dtype=np.float32)
            no_contrast_raw = np.concatenate(no_contrast_list, axis=0) if no_contrast_list else np.empty((0,), dtype=np.float32)
            embeddings = np.concatenate(emb_list, axis=0) if emb_list else np.empty((0, 0), dtype=np.float32)
            cic_full = np.concatenate(cic_list, axis=0) if cic_list else np.empty((0, 4), dtype=np.float32)

            malicious_indices = set(_load_malicious_indices(data_dir))
            labels_full = np.zeros(total_nodes, dtype=np.int32)
            for idx in malicious_indices:
                if 0 <= idx < total_nodes:
                    labels_full[idx] = 1
            keep_mask = (np.arange(total_nodes) >= skip_benign) | (labels_full == 1)

            cic_kept = cic_full[keep_mask]
            full_kept_raw = full_raw[keep_mask]
            no_mask_kept_raw = no_mask_raw[keep_mask]
            no_contrast_kept_raw = no_contrast_raw[keep_mask]
            embeddings_kept = embeddings[keep_mask]
            labels_kept = labels_full[keep_mask]
            cic_scores_kept = cic_kept

            full_kept = _apply_ranking_adjustment(full_kept_raw, cic_kept)
            no_mask_kept = _apply_ranking_adjustment(no_mask_kept_raw, cic_kept)
            no_contrast_kept = _apply_ranking_adjustment(no_contrast_kept_raw, cic_kept)

            pred_scores_full = full_kept
            pred_scores_variants = {
                "no_masking": no_mask_kept,
                "no_contrastive": no_contrast_kept,
                "no_ranking": full_kept_raw,
            }
        else:
            print(f"[WARN] 未找到模型检查点 (seed={seed_value})，模块消融将跳过，仅执行CIC相关消融。")

        print(f"\n[3/3] 运行消融实验 (seed={seed_value})...")
        results = run_full_ablation_study(
            cic_scores_4d=cic_scores_kept,
            labels=labels_kept,
            pred_scores_full=pred_scores_full,
            pred_scores_variants=pred_scores_variants,
            embeddings=embeddings_kept,
            fusion_top_k=int(args.fusion_top_k),
            output_dir=seed_output_dir,
        )

        output_path = os.path.join(seed_output_dir, f"ablation_results_{args.dataset}_s{seed_value}.json")
        save_evaluation_results(results, output_path)
        print(f"[OK] 结果已保存: {output_path}")
        metrics = _extract_ablation_metrics(results)
        readable_path = os.path.join(
            seed_output_dir,
            f"ablation_summary_{args.dataset}_s{seed_value}_readable.txt",
        )
        _write_ablation_readable(
            readable_path,
            {
                "dataset": args.dataset,
                "seed": seed_value,
                "checkpoint": ckpt_path,
            },
            metrics,
        )
        seed_summaries.append(
            {
                "seed": seed_value,
                "checkpoint": ckpt_path,
                "metrics": metrics,
            }
        )

    if seed_summaries:
        summary = _summarize_metrics([item.get("metrics") for item in seed_summaries])
        dataset_output_dir = os.path.join(args.output_dir, args.dataset)
        summary_path = os.path.join(dataset_output_dir, f"ablation_summary_{args.dataset}_all_seeds.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": args.dataset,
                    "num_seeds": len(seed_summaries),
                    "seeds": [item.get("seed") for item in seed_summaries],
                    **summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[OK] 汇总已保存: {summary_path}")
        readable_path = os.path.join(
            dataset_output_dir,
            f"ablation_summary_{args.dataset}_all_seeds_readable.txt",
        )
        _write_ablation_summary_readable(
            readable_path,
            {
                "dataset": args.dataset,
                "num_seeds": len(seed_summaries),
                "seeds": [item.get("seed") for item in seed_summaries],
            },
            summary,
        )


if __name__ == '__main__':
    main()
