"""
Evaluation entrypoint for MAGIC.

- Requires CUDA (GPU) to run.
- Writes evaluation metrics to disk (no console output).
"""

import os
import sys
import json
import torch
import warnings
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.loaddata import (
    load_batch_level_dataset,
    load_entity_level_dataset,
    load_entity_level_dataset_with_cic,
    load_metadata,
    load_metadata_with_cic,
)
from model.autoencoder import build_model
from utils.poolers import Pooling
from utils.utils import set_random_seed
from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn
from sklearn.metrics import roc_auc_score
from utils.config import build_args
warnings.filterwarnings('ignore')


DATA_COMPASS_DIR = os.path.join("results", "data_compass")


def _require_cuda(device_index: int) -> torch.device:
    if device_index < 0:
        raise RuntimeError("GPU is required. Please pass --device 0 (or another GPU index).")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required but CUDA is not available.")
    return torch.device(f"cuda:{device_index}")


def _dataset_dir(base_dir: str, dataset_name: str) -> str:
    return os.path.join(base_dir, dataset_name)


def _checkpoint_path(dataset_name: str, seed: int = None) -> str:
    if seed is not None:
        return os.path.join("checkpoints", dataset_name, f"checkpoint-{dataset_name}_s{seed}.pt")
    return os.path.join("checkpoints", dataset_name, f"checkpoint-{dataset_name}.pt")


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


def _seed_tag(seed: int) -> str:
    return f"s{seed}"




def _write_json(path: str, payload: dict, comment: str = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if comment:
            payload = dict(payload)
            payload.setdefault("_comment", comment)
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _summarize_seed_metrics(metrics_list: list, keys: list) -> tuple:
    means = {}
    stds = {}
    for key in keys:
        values = [m.get(key) for m in metrics_list if isinstance(m, dict) and key in m]
        if not values:
            means[key] = float("nan")
            stds[key] = float("nan")
            continue
        arr = np.array(values, dtype=float)
        means[key] = float(np.nanmean(arr))
        stds[key] = float(np.nanstd(arr))
    return means, stds


def main(main_args):
    device = _require_cuda(int(main_args.device))
    dataset_name = main_args.dataset
    num_seeds = max(1, int(getattr(main_args, "num_seeds", 1)))
    seed_start = int(getattr(main_args, "seed_start", 0))
    results_dir = _dataset_dir(DATA_COMPASS_DIR, dataset_name)
    if dataset_name in ['streamspot', 'wget']:
        main_args.num_hidden = 256
        main_args.num_layers = 4
    else:
        main_args.num_hidden = 64
        main_args.num_layers = 3

    if dataset_name == 'streamspot' or dataset_name == 'wget':
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        pooler = Pooling(main_args.pooling)
        seed_checkpoints = _discover_seed_checkpoints(dataset_name)
        if seed_checkpoints:
            seed_total = len(seed_checkpoints)
            seed_iter = tqdm(seed_checkpoints, desc="Seeds", leave=True) if seed_total > 1 else seed_checkpoints
            knn_repeat = 1 if seed_total > 1 else 100
            print(f"[EVAL] dataset={dataset_name} mode=batch found {seed_total} seed checkpoints")
            for seed_idx, (seed_value, ckpt_path) in enumerate(seed_iter):
                seed_tag = _seed_tag(seed_value)
                set_random_seed(seed_value)
                print(f"[EVAL] seed={seed_value} ({seed_idx+1}/{seed_total}) loading {ckpt_path}")

                model = build_model(main_args)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model = model.to(device)
                model.eval()

                extra = {
                    "dataset": dataset_name,
                    "device": str(device),
                    "seed": int(seed_value),
                    "checkpoint": ckpt_path,
                }
                log_path = os.path.join(results_dir, f"eval_metrics_{dataset_name}_{seed_tag}.json")
                test_auc, test_std = batch_level_evaluation(
                    model,
                    pooler,
                    device,
                    ['knn'],
                    dataset_name,
                    main_args.n_dim,
                    main_args.e_dim,
                    log_path=log_path,
                    log_extra=extra,
                    show_progress=True,
                    knn_repeat=knn_repeat,
                    verbose=False,
                )
                print(f"[EVAL] seed={seed_value} done (auc={test_auc:.6f}, std={test_std:.6f})")
        else:
            print(f"[EVAL] dataset={dataset_name} mode=batch seeds={num_seeds}")
            seed_iter = tqdm(range(num_seeds), desc="Seeds", leave=True) if num_seeds > 1 else range(num_seeds)
            knn_repeat = 1 if num_seeds > 1 else 100
            for idx in seed_iter:
                seed_value = seed_start + idx
                seed_index = idx + 1
                seed_tag = _seed_tag(seed_index)
                set_random_seed(seed_value)
                print(f"[EVAL] seed={seed_value} ({seed_index}/{num_seeds}) start")

                model = build_model(main_args)
                model.load_state_dict(torch.load(_checkpoint_path(dataset_name), map_location=device))
                model = model.to(device)
                model.eval()

                extra = {
                    "dataset": dataset_name,
                    "device": str(device),
                    "seed": int(seed_value),
                    "seed_index": int(seed_index),
                }
                log_path = os.path.join(results_dir, f"eval_metrics_{dataset_name}_{seed_tag}.json")
                test_auc, test_std = batch_level_evaluation(
                    model,
                    pooler,
                    device,
                    ['knn'],
                    dataset_name,
                    main_args.n_dim,
                    main_args.e_dim,
                    log_path=log_path,
                    log_extra=extra,
                    show_progress=True,
                    knn_repeat=knn_repeat,
                    verbose=False,
                )
                print(f"[EVAL] seed={seed_value} done (auc={test_auc:.6f}, std={test_std:.6f})")
    else:
        # 自动发现所有训练好的 seed checkpoints
        seed_checkpoints = _discover_seed_checkpoints(dataset_name)
        if not seed_checkpoints:
            print(f"[EVAL] No seed checkpoints found for {dataset_name}. Run training first.")
            return
        
        print(f"[EVAL] dataset={dataset_name} mode=entity found {len(seed_checkpoints)} seed checkpoints")
        
        metadata = load_metadata_with_cic(dataset_name)
        use_cic = bool(getattr(main_args, "use_cic", False)) and bool(metadata.get("has_cic_scores", False))
        add_cic_to_features = use_cic and bool(getattr(main_args, "cic_as_node_feature", False))

        main_args.n_dim = metadata['node_feature_dim'] + (4 if add_cic_to_features else 0)
        main_args.e_dim = metadata['edge_feature_dim']
        
        # 安全解构 malicious 信息
        malicious_info = metadata.get('malicious')
        if malicious_info is None:
            malicious = []
        elif isinstance(malicious_info, (list, tuple)) and len(malicious_info) >= 1:
            malicious = malicious_info[0] if isinstance(malicious_info[0], (list, np.ndarray)) else list(malicious_info)
        else:
            malicious = list(malicious_info) if hasattr(malicious_info, '__iter__') else []
        
        n_train = metadata['n_train']
        n_test = metadata['n_test']

        seed_metrics = []
        seed_total = len(seed_checkpoints)
        seed_iter = tqdm(seed_checkpoints, desc="Seeds", leave=True) if seed_total > 1 else seed_checkpoints
        use_cache = seed_total <= 1
        cache_path = os.path.join("eval_result", dataset_name, f"distance_save_{dataset_name}.pkl")
        if seed_total > 1 and os.path.exists(cache_path):
            os.unlink(cache_path)

        for seed_idx, (seed_value, ckpt_path) in enumerate(seed_iter):
            seed_tag = _seed_tag(seed_value)
            set_random_seed(seed_value)
            print(f"[EVAL] seed={seed_value} ({seed_idx+1}/{seed_total}) loading {ckpt_path}")
            
            # 为每个 seed 重新加载模型
            model = build_model(main_args)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model = model.to(device)
            model.eval()

            extra = {
                "dataset": dataset_name,
                "device": str(device),
                "seed": int(seed_value),
                "checkpoint": ckpt_path,
                "use_cic": bool(use_cic),
                "cic_as_node_feature": bool(add_cic_to_features),
            }

            with torch.no_grad():
                x_train = []
                train_iter = (
                    tqdm(range(n_train), desc=f"Seed {seed_idx+1}/{seed_total} train", leave=False)
                    if n_train > 1
                    else range(n_train)
                )
                for i in train_iter:
                    loader = load_entity_level_dataset_with_cic if use_cic else load_entity_level_dataset
                    g = loader(dataset_name, 'train', i).to(device)
                    if add_cic_to_features and 'cic_scores' in g.ndata and 'attr' in g.ndata:
                        cic = g.ndata['cic_scores'].to(device=g.ndata['attr'].device, dtype=g.ndata['attr'].dtype)
                        g.ndata['attr'] = torch.cat([g.ndata['attr'], cic], dim=-1)
                    x_train.append(model.embed(g).cpu().numpy())
                    del g
                x_train = np.concatenate(x_train, axis=0)
                print(f"[EVAL] seed={seed_value} train_embeddings={x_train.shape}")
                skip_benign = 0
                x_test = []
                fusion_scores = []  # optional: CIC + contrastive fused scores, aligned with x_test order
                if use_cic:
                    from model.fusion import AnomalyScorer
                    from model.contrastive import NodeLevelContrastive

                    hidden_dim = getattr(model, 'output_hidden_dim', getattr(model, 'num_hidden', 64))
                    node_contrast = NodeLevelContrastive(hidden_dim=hidden_dim).to(device)
                    fusion = AnomalyScorer(n_sources=2)  # CIC + contrastive
                    fusion = fusion.to(device)
                test_iter = (
                    tqdm(range(n_test), desc=f"Seed {seed_idx+1}/{seed_total} test", leave=False)
                    if n_test > 1
                    else range(n_test)
                )
                for i in test_iter:
                    loader = load_entity_level_dataset_with_cic if use_cic else load_entity_level_dataset
                    g = loader(dataset_name, 'test', i).to(device)
                    # Exclude training samples from the test set
                    if i != n_test - 1:
                        skip_benign += g.number_of_nodes()
                    if add_cic_to_features and 'cic_scores' in g.ndata and 'attr' in g.ndata:
                        cic = g.ndata['cic_scores'].to(device=g.ndata['attr'].device, dtype=g.ndata['attr'].dtype)
                        g.ndata['attr'] = torch.cat([g.ndata['attr'], cic], dim=-1)
                    emb = model.embed(g)
                    x_test.append(emb.cpu().numpy())

                    if use_cic:
                        cic_scores = g.ndata.get('cic_scores')
                        if cic_scores is None:
                            fusion_scores.append(np.zeros(g.num_nodes(), dtype=np.float32))
                        else:
                            if isinstance(emb, tuple):
                                emb = emb[0]
                            contrast_score = node_contrast.anomaly_score(
                                emb, cic_scores, threshold=float(getattr(main_args, "cic_anomaly_threshold", 0.5))
                            )
                            fused = fusion.compute_anomaly_score(cic_scores, contrastive_score=contrast_score, recon_error=None)
                            fusion_scores.append(fused.detach().cpu().numpy())
                    del g
                x_test = np.concatenate(x_test, axis=0)
                fusion_scores = np.concatenate(fusion_scores, axis=0) if (use_cic and fusion_scores) else None
                print(f"[EVAL] seed={seed_value} test_embeddings={x_test.shape} skip_benign={skip_benign}")

                n = x_test.shape[0]
                y_test = np.zeros(n)
                y_test[malicious] = 1.0

                # Exclude training samples from the test set
                test_idx = []
                for i in range(x_test.shape[0]):
                    if i >= skip_benign or y_test[i] == 1.0:
                        test_idx.append(i)
                result_x_test = x_test[test_idx]
                result_y_test = y_test[test_idx]
                result_fusion = fusion_scores[test_idx] if fusion_scores is not None else None
                del x_test, y_test
                print(f"[EVAL] seed={seed_value} knn_input train={x_train.shape} test={result_x_test.shape}")
                log_path = os.path.join(results_dir, f"eval_metrics_{dataset_name}_{seed_tag}.json")
                test_auc, test_std, metrics, _ = evaluate_entity_level_using_knn(
                    dataset_name,
                    x_train,
                    result_x_test,
                    result_y_test,
                    log_path=log_path,
                    log_extra=extra,
                    verbose=False,
                )
                if isinstance(metrics, dict):
                    seed_metrics.append(metrics)
                print(f"[EVAL] seed={seed_value} done (auc={test_auc:.6f}, std={test_std:.6f})")

                if result_fusion is not None:
                    fusion_auc = roc_auc_score(result_y_test, result_fusion)
                    _write_json(
                        os.path.join(results_dir, f"eval_fusion_{dataset_name}_{seed_tag}.json"),
                        {
                            **extra,
                            "fusion_auc": float(fusion_auc),
                        },
                        comment="Fusion-based evaluation metrics (CIC + contrastive).",
                    )
            # 注意：不再生成 eval_summary_{dataset_name}_{seed_tag}.json，因为 eval_metrics 已包含完整信息
        if seed_metrics:
            summary_keys = [
                "auc",
                "pr_auc",
                "f1",
                "precision",
                "recall",
                "tpr_at_fpr_0.001",
                "tpr_at_fpr_0.01",
                "fpr_at_tpr_0.95",
                "fpr_at_tpr_0.99",
            ]
            means, stds = _summarize_seed_metrics(seed_metrics, summary_keys)
            _write_json(
                os.path.join(results_dir, f"eval_summary_{dataset_name}_all_seeds.json"),
                {
                    "dataset": dataset_name,
                    "device": str(device),
                    "num_seeds": int(seed_total),
                    "seeds": [int(seed) for seed, _ in seed_checkpoints],
                    "use_cic": bool(use_cic),
                    "cic_as_node_feature": bool(add_cic_to_features),
                    **means,
                    "metrics_std": stds,
                },
                comment="Summary metrics averaged across seeds.",
            )
    return


if __name__ == '__main__':
    args = build_args()
    main(args)
