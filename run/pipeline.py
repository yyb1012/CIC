"""
End-to-end pipeline for MAGIC (DARPA E5: theia/cadets/clearscope).

Stages:
  1) preprocess: trace_parser -> CIC-enhanced graphs (metadata_cic.json, *cic*.pkl)
  2) train:      train GMAE autoencoder (masking/reconstruction)
  3) eval:       node-level fusion score + optimal threshold search
  4) explain:    build minimal explanation subgraphs for top anomalies

Notes:
  - Project and datasets can be in different roots. `--data_dir` can be either:
      (a) dataset directory, e.g. /hy-tmp/data/theia
      (b) dataset root,     e.g. /hy-tmp/data  (pipeline will append /{dataset})
  - CIC is treated as "always on" when available; if CIC files are missing, it falls back gracefully.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle as pkl
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _print_header(title: str) -> None:
    line = "=" * 72
    print("\n" + line)
    print(title)
    print(line)


def _looks_like_dataset_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    markers = [
        os.path.join(path, "train.pkl"),
        os.path.join(path, "test.pkl"),
        os.path.join(path, "metadata_cic.json"),
        os.path.join(path, "malicious.pkl"),
        os.path.join(path, "entities.pkl"),
    ]
    if any(os.path.exists(m) for m in markers):
        return True
    try:
        import glob

        if glob.glob(os.path.join(path, "ta1-*.json")):
            return True
    except Exception:
        pass
    return False


def resolve_dataset_dir(data_dir: str, dataset: str) -> Tuple[str, str]:
    """
    Resolve dataset directory and (optionally) set DATA_ROOT for other loaders.
    Returns:
        dataset_dir, data_root
    """
    if not data_dir:
        raise ValueError("data_dir is required")

    expanded = os.path.abspath(os.path.expanduser(data_dir))
    if _looks_like_dataset_dir(expanded):
        dataset_dir = expanded
        data_root = os.path.dirname(dataset_dir)
    else:
        candidate = os.path.join(expanded, dataset)
        if _looks_like_dataset_dir(candidate):
            dataset_dir = candidate
            data_root = expanded
        else:
            # Fall back: treat as dataset directory (even if files not present yet)
            dataset_dir = candidate if os.path.isdir(candidate) else expanded
            data_root = os.path.dirname(dataset_dir)

    if dataset_dir and os.path.isdir(dataset_dir):
        os.environ.setdefault("DATA_ROOT", data_root)

    return dataset_dir, data_root


def _ensure_dataset_subdir(base_dir: str, dataset: str) -> str:
    base_dir = os.path.normpath(base_dir)
    if os.path.basename(base_dir) == dataset:
        return base_dir
    return os.path.join(base_dir, dataset)


def _resolve_checkpoint_dir(base_dir: str, dataset: str) -> str:
    candidate = _ensure_dataset_subdir(base_dir, dataset)
    compat = os.path.join(candidate, f"checkpoint-{dataset}.pt")
    legacy = os.path.join(base_dir, f"checkpoint-{dataset}.pt")
    if os.path.exists(compat) or not os.path.exists(legacy):
        return candidate
    return base_dir


def _device_from_string(device: str) -> Tuple[torch.device, int]:
    d = str(device).strip().lower()
    if d in {"cpu", "-1"}:
        return torch.device("cpu"), -1
    if d.startswith("cuda"):
        if ":" in d:
            try:
                idx = int(d.split(":", 1)[1])
            except ValueError:
                idx = 0
        else:
            idx = 0
        if torch.cuda.is_available():
            return torch.device(f"cuda:{idx}"), idx
        return torch.device("cpu"), -1
    try:
        idx = int(d)
        if idx >= 0 and torch.cuda.is_available():
            return torch.device(f"cuda:{idx}"), idx
        return torch.device("cpu"), -1
    except ValueError:
        return torch.device("cpu"), -1


def _build_default_model_args() -> argparse.Namespace:
    """
    utils/config.py:build_args() parses sys.argv, which would include pipeline flags.
    Build a clean default args object by temporarily clearing argv.
    """
    from utils.config import build_args as _build_args

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        return _build_args()
    finally:
        sys.argv = old_argv


def _configure_model_args(
    *,
    base_args: argparse.Namespace,
    dataset: str,
    epochs: int,
    device_index: int,
    metadata: Dict[str, Any],
) -> Tuple[argparse.Namespace, bool, bool]:
    # Match train.py / eval.py defaults for E3 datasets
    if dataset in {"streamspot", "wget"}:
        base_args.num_hidden = 256
        base_args.num_layers = 4
    else:
        base_args.num_hidden = 64
        base_args.num_layers = 3
    base_args.max_epoch = int(epochs)
    base_args.dataset = dataset
    base_args.device = int(device_index)

    use_cic = bool(getattr(base_args, "use_cic", False)) and bool(metadata.get("has_cic_scores", False))
    add_cic_to_features = use_cic and bool(getattr(base_args, "cic_as_node_feature", False))

    base_args.n_dim = int(metadata.get("node_feature_dim", 0)) + (4 if add_cic_to_features else 0)
    base_args.e_dim = int(metadata.get("edge_feature_dim", 0))
    if base_args.n_dim <= 0 or base_args.e_dim <= 0:
        raise ValueError(f"Invalid feature dims: n_dim={base_args.n_dim}, e_dim={base_args.e_dim}")

    return base_args, use_cic, add_cic_to_features


def _load_malicious_indices(dataset_dir: str, metadata: Dict[str, Any]) -> List[int]:
    malicious = metadata.get("malicious")
    if isinstance(malicious, (list, tuple)) and len(malicious) == 2 and isinstance(malicious[0], (list, tuple)):
        return [int(x) for x in malicious[0]]
    if isinstance(malicious, list):
        return [int(x) for x in malicious]
    if isinstance(malicious, dict) and "nodes" in malicious:
        return [int(x) for x in malicious.get("nodes", [])]

    # Fallback to malicious.pkl (older caches)
    mpath = os.path.join(dataset_dir, "malicious.pkl")
    if not os.path.exists(mpath):
        return []
    with open(mpath, "rb") as f:
        data = pkl.load(f)
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], (list, tuple)):
        return [int(x) for x in data[0]]
    if isinstance(data, list):
        return [int(x) for x in data]
    if isinstance(data, dict) and "nodes" in data:
        return [int(x) for x in data.get("nodes", [])]
    return []


def _unwrap_cic_meta_map(raw: Any, key: str) -> Dict[Any, Any]:
    if isinstance(raw, dict):
        inner = raw.get(key)
        if isinstance(inner, dict):
            return inner
        return raw
    return {}


def _build_node_maps_from_raw(
    dataset_dir: str,
    test_index: int,
    name_by_uuid: Dict[Any, Any],
    type_by_uuid: Dict[Any, Any],
) -> Tuple[Dict[int, Any], Dict[int, Any]]:
    raw_path = os.path.join(dataset_dir, "test.pkl")
    if not os.path.exists(raw_path):
        return {}, {}

    try:
        with open(raw_path, "rb") as f:
            raw_data = pkl.load(f)
    except Exception:
        return {}, {}

    if isinstance(raw_data, (list, tuple)):
        if test_index < 0 or test_index >= len(raw_data):
            return {}, {}
        raw_graph = raw_data[test_index]
    elif isinstance(raw_data, dict):
        raw_graph = raw_data
    else:
        return {}, {}

    try:
        import networkx as nx

        g_nx = nx.node_link_graph(raw_graph)
    except Exception:
        return {}, {}

    def _coerce_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bytes):
            for enc in ("utf-8", "gbk", "latin-1"):
                try:
                    return value.decode(enc)
                except Exception:
                    continue
            return value.decode("utf-8", errors="replace")
        return str(value)

    id_to_name: Dict[int, Any] = {}
    id_to_type: Dict[int, Any] = {}

    for nid, attrs in g_nx.nodes(data=True):
        node_id = int(nid)
        uuid = attrs.get("uuid")

        name = None
        for key in ("filename", "exe_path", "cmdline", "remote_address", "local_address", "name"):
            val = attrs.get(key)
            if val:
                name = val
                break
        if not name and uuid:
            name = name_by_uuid.get(uuid)
        if not name and uuid:
            name = uuid
        name = _coerce_text(name)
        if name:
            id_to_name[node_id] = name

        type_name = None
        if uuid:
            type_name = type_by_uuid.get(uuid)
        if not type_name:
            type_name = attrs.get("type_name")
        type_name = _coerce_text(type_name)
        if type_name:
            id_to_type[node_id] = type_name

    return id_to_name, id_to_type


# ============================================================================
# Stage 1: preprocess
# ============================================================================


def run_preprocess(dataset_dir: str, dataset: str) -> bool:
    _print_header("Stage 1: preprocess (trace_parser + CIC)")

    train_pkl = os.path.join(dataset_dir, "train.pkl")
    test_pkl = os.path.join(dataset_dir, "test.pkl")
    if os.path.exists(train_pkl) and os.path.exists(test_pkl):
        print(f"[SKIP] trace_parser: found `{train_pkl}` and `{test_pkl}`")
    else:
        print("[RUN ] trace_parser")
        parser_script = os.path.join(PROJECT_ROOT, "utils", "trace_parser.py")
        cmd = f'python "{parser_script}" --dataset {dataset} --data_dir "{dataset_dir}"'
        ret = os.system(cmd)
        if ret != 0:
            print(f"[ERR ] trace_parser failed (exit={ret})")
            return False

    meta_cic = os.path.join(dataset_dir, "metadata_cic.json")
    if os.path.exists(meta_cic):
        print(f"[SKIP] CIC enhance: found `{meta_cic}`")
    else:
        print("[RUN ] CIC enhance: preload_entity_level_dataset_with_cic")
        from utils.loaddata import preload_entity_level_dataset_with_cic

        preload_entity_level_dataset_with_cic(dataset_dir, compute_cic=True)
        if not os.path.exists(meta_cic):
            print(f"[ERR ] CIC enhance did not create `{meta_cic}`")
            return False

    cic_scores_pkl = os.path.join(dataset_dir, "cic_scores.pkl")
    if os.path.exists(cic_scores_pkl):
        print(f"[OK  ] CIC scores: found `{cic_scores_pkl}`")
    else:
        # If CIC enhance ran with compute_cic=True, this should already exist; keep a safe fallback.
        print("[RUN ] CIC scores: compute_and_save_cic_scores")
        from utils.cic_invariants import compute_and_save_cic_scores

        compute_and_save_cic_scores(dataset_dir)
        if not os.path.exists(cic_scores_pkl):
            print(f"[WARN] CIC scores still missing: `{cic_scores_pkl}`")

    return True


# ============================================================================
# Stage 2: train
# ============================================================================


def run_train(
    *,
    dataset_dir: str,
    dataset: str,
    device: torch.device,
    device_index: int,
    epochs: int,
    checkpoint_dir: str,
    log_interval: int,
    save_interval: int,
    seed: int,
) -> bool:
    _print_header("Stage 2: train (GMAE autoencoder)")

    os.makedirs(checkpoint_dir, exist_ok=True)

    from utils.loaddata import load_metadata_with_cic, load_entity_level_dataset, load_entity_level_dataset_with_cic
    from utils.utils import create_optimizer, set_random_seed
    from model.autoencoder import build_model

    set_random_seed(int(seed))

    metadata = load_metadata_with_cic(dataset_dir)
    model_args = _build_default_model_args()
    model_args, use_cic, add_cic = _configure_model_args(
        base_args=model_args, dataset=dataset, epochs=epochs, device_index=device_index, metadata=metadata
    )

    n_train = int(metadata.get("n_train", 0))
    if n_train <= 0:
        print("[ERR ] Invalid metadata: n_train <= 0")
        return False

    loader = load_entity_level_dataset_with_cic if use_cic else load_entity_level_dataset

    model = build_model(model_args).to(device)
    optimizer = create_optimizer(model_args.optimizer, model, model_args.lr, model_args.weight_decay)

    print(f"[INFO] dataset_dir: {dataset_dir}")
    print(f"[INFO] use_cic={use_cic} add_cic_to_features={add_cic}")
    print(f"[INFO] n_dim={model_args.n_dim} e_dim={model_args.e_dim} n_train={n_train} epochs={epochs}")

    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_loss = 0.0

        for i in range(n_train):
            g = loader(dataset_dir, "train", i).to(device)
            if add_cic and "cic_scores" in g.ndata and "attr" in g.ndata:
                cic = g.ndata["cic_scores"].to(device=g.ndata["attr"].device, dtype=g.ndata["attr"].dtype)
                g.ndata["attr"] = torch.cat([g.ndata["attr"], cic], dim=-1)

            loss = model(g)
            loss = loss / float(n_train)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())
            del g

        if epoch % int(log_interval) == 0 or epoch == 1 or epoch == int(epochs):
            print(f"[INFO] epoch {epoch:4d}/{epochs} loss={epoch_loss:.6f}")

        if epoch % int(save_interval) == 0 or epoch == int(epochs):
            ckpt_path = os.path.join(checkpoint_dir, f"model_{dataset}_epoch{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": epoch_loss,
                    "dataset": dataset,
                    "dataset_dir": dataset_dir,
                    "timestamp": datetime.now().isoformat(),
                },
                ckpt_path,
            )
            print(f"[OK  ] saved checkpoint: {ckpt_path}")

    # Save final (pipeline format) + compatibility checkpoint (train.py/eval.py format)
    final_path = os.path.join(checkpoint_dir, f"model_{dataset}_final.pt")
    torch.save(
        {
            "epoch": int(epochs),
            "model_state_dict": model.state_dict(),
            "dataset": dataset,
            "dataset_dir": dataset_dir,
            "timestamp": datetime.now().isoformat(),
        },
        final_path,
    )
    compat_path = os.path.join(checkpoint_dir, f"checkpoint-{dataset}.pt")
    torch.save(model.state_dict(), compat_path)
    print(f"[OK  ] saved final: {final_path}")
    print(f"[OK  ] saved compat: {compat_path}")
    return True


# ============================================================================
# Stage 3: eval
# ============================================================================


@torch.no_grad()
def _compute_graph_scores(
    *,
    model: torch.nn.Module,
    g: Any,
    anomaly_scorer: Optional[torch.nn.Module],
    node_contrastive: Optional[torch.nn.Module],
    anomaly_threshold: float,
) -> Dict[str, torch.Tensor]:
    # embeddings
    emb = model.embed(g)
    if isinstance(emb, tuple):
        emb = emb[0]

    cic_scores = g.ndata.get("cic_scores")
    if cic_scores is None:
        cic_scores = torch.zeros((g.num_nodes(), 4), device=g.device, dtype=torch.float32)
    else:
        cic_scores = cic_scores.to(device=g.device, dtype=torch.float32)

    recon = None
    if hasattr(model, "node_reconstruction_error"):
        recon = model.node_reconstruction_error(g)

    contrast = None
    if node_contrastive is not None and hasattr(node_contrastive, "anomaly_score"):
        contrast = node_contrastive.anomaly_score(emb, cic_scores, threshold=float(anomaly_threshold))

    fused = None
    if anomaly_scorer is not None and hasattr(anomaly_scorer, "compute_anomaly_score"):
        fused = anomaly_scorer.compute_anomaly_score(cic_scores, contrastive_score=contrast, recon_error=recon)
    else:
        # Safe fallback: CIC total only
        fused = 1.0 - (1.0 - 0.25 * cic_scores).prod(dim=1)

    fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    return {
        "embeddings": emb,
        "cic_scores": cic_scores,
        "recon_error": recon if recon is not None else torch.zeros(g.num_nodes(), device=g.device),
        "contrastive_score": contrast if contrast is not None else torch.zeros(g.num_nodes(), device=g.device),
        "fused_score": fused,
    }


def run_eval(
    *,
    dataset_dir: str,
    dataset: str,
    device: torch.device,
    device_index: int,
    checkpoint_dir: str,
    results_dir: str,
    anomaly_threshold: float,
    epochs_for_shape: int,
) -> bool:
    _print_header("Stage 3: eval (node-level fusion + threshold search)")

    os.makedirs(results_dir, exist_ok=True)

    from utils.loaddata import load_metadata_with_cic, load_entity_level_dataset, load_entity_level_dataset_with_cic, load_cic_metadata
    from model.autoencoder import build_model
    from utils.utils import set_random_seed
    from model.fusion import AnomalyScorer
    from model.contrastive import NodeLevelContrastive
    from experiments.evaluation import EntityLevelEvaluator

    set_random_seed(0)

    metadata = load_metadata_with_cic(dataset_dir)
    model_args = _build_default_model_args()
    model_args, use_cic, add_cic = _configure_model_args(
        base_args=model_args, dataset=dataset, epochs=epochs_for_shape, device_index=device_index, metadata=metadata
    )

    model = build_model(model_args).to(device)

    # Prefer compatibility checkpoint, then pipeline final, then seed-specific
    compat_path = os.path.join(checkpoint_dir, f"checkpoint-{dataset}.pt")
    final_path = os.path.join(checkpoint_dir, f"model_{dataset}_final.pt")
    loaded_ckpt = None
    
    if os.path.exists(compat_path):
        model.load_state_dict(torch.load(compat_path, map_location=device))
        print(f"[OK  ] loaded model: {compat_path}")
        loaded_ckpt = compat_path
    elif os.path.exists(final_path):
        ckpt = torch.load(final_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[OK  ] loaded model: {final_path}")
        loaded_ckpt = final_path
    else:
        # Try seed-specific checkpoints (s0, s1, ...)
        for seed_idx in range(10):
            seed_path = os.path.join(checkpoint_dir, f"checkpoint-{dataset}_s{seed_idx}.pt")
            if os.path.exists(seed_path):
                model.load_state_dict(torch.load(seed_path, map_location=device))
                print(f"[OK  ] loaded model: {seed_path}")
                loaded_ckpt = seed_path
                break
    
    if loaded_ckpt is None:
        print(f"[ERR ] missing checkpoint: `{compat_path}` or `{final_path}` or `checkpoint-{dataset}_s*.pt`")
        return False

    model.eval()

    n_test = int(metadata.get("n_test", 0))
    if n_test <= 0:
        print("[ERR ] Invalid metadata: n_test <= 0")
        return False

    malicious_indices = set(_load_malicious_indices(dataset_dir, metadata))
    cic_meta = load_cic_metadata(dataset_dir)
    names_uuid_map = _unwrap_cic_meta_map(cic_meta.get("names"), "id_nodename_map")
    types_uuid_map = _unwrap_cic_meta_map(cic_meta.get("types"), "id_nodetype_map")

    names_map, types_map = _build_node_maps_from_raw(
        dataset_dir,
        n_test - 1,
        names_uuid_map,
        types_uuid_map,
    )

    loader = load_entity_level_dataset_with_cic if use_cic else load_entity_level_dataset
    node_contrast = NodeLevelContrastive(hidden_dim=int(getattr(model, "output_hidden_dim", 64))).to(device)
    anomaly_scorer = AnomalyScorer(n_sources=3).to(device)

    scores_all: List[np.ndarray] = []
    cic_all: List[np.ndarray] = []
    total_nodes = 0
    skip_benign = 0

    last_graph_payload: Dict[str, Any] = {}

    for i in range(n_test):
        g = loader(dataset_dir, "test", i).to(device)
        if add_cic and "cic_scores" in g.ndata and "attr" in g.ndata:
            cic = g.ndata["cic_scores"].to(device=g.ndata["attr"].device, dtype=g.ndata["attr"].dtype)
            g.ndata["attr"] = torch.cat([g.ndata["attr"], cic], dim=-1)

        payload = _compute_graph_scores(
            model=model,
            g=g,
            anomaly_scorer=anomaly_scorer,
            node_contrastive=node_contrast,
            anomaly_threshold=anomaly_threshold,
        )

        score_np = payload["fused_score"].detach().cpu().numpy()
        cic_np = payload["cic_scores"].detach().cpu().numpy()
        scores_all.append(score_np)
        cic_all.append(cic_np)

        total_nodes += int(g.num_nodes())
        if i != n_test - 1:
            skip_benign += int(g.num_nodes())
        else:
            last_graph_payload = {
                "test_index": i,
                "n_nodes": int(g.num_nodes()),
                "scores": score_np,
                "cic_scores": cic_np,
                "names_map": names_map,
                "types_map": types_map,
            }

        del g

    scores_full = np.concatenate(scores_all, axis=0) if scores_all else np.empty((0,), dtype=np.float32)
    cic_full = np.concatenate(cic_all, axis=0) if cic_all else np.empty((0, 4), dtype=np.float32)

    labels_full = np.zeros(total_nodes, dtype=np.int32)
    for idx in malicious_indices:
        if 0 <= int(idx) < total_nodes:
            labels_full[int(idx)] = 1

    keep_mask = (np.arange(total_nodes) >= int(skip_benign)) | (labels_full == 1)
    scores_kept = scores_full[keep_mask]
    labels_kept = labels_full[keep_mask]

    evaluator = EntityLevelEvaluator()
    base = evaluator.evaluate(scores_kept, labels_kept)
    best_thr, best = evaluator.find_optimal_threshold(scores_kept, labels_kept)

    results_path = os.path.join(results_dir, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset,
                "dataset_dir": dataset_dir,
                "n_test_graphs": n_test,
                "total_nodes_all_test_graphs": int(total_nodes),
                "kept_nodes_for_eval": int(labels_kept.shape[0]),
                "malicious_total": int(labels_full.sum()),
                "malicious_kept": int(labels_kept.sum()),
                "use_cic": bool(use_cic),
                "cic_as_node_feature": bool(add_cic),
                "basic_result": base.to_dict(),
                "optimal_threshold": float(best_thr),
                "optimal_result": best.to_dict(),
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[OK  ] saved: {results_path}")

    thr_path = os.path.join(results_dir, "optimal_threshold.json")
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": float(best_thr)}, f, indent=2)
    print(f"[OK  ] saved: {thr_path}")

    # Save for explain stage (focus on last test graph)
    scores_path = os.path.join(results_dir, "anomaly_scores.pkl")
    last_labels = labels_full[int(skip_benign) :] if int(skip_benign) <= total_nodes else np.zeros((0,), np.int32)
    with open(scores_path, "wb") as f:
        pkl.dump(
            {
                "dataset": dataset,
                "dataset_dir": dataset_dir,
                "threshold": float(best_thr),
                "eval_keep_mask": keep_mask,
                "eval_scores": scores_kept,
                "eval_labels": labels_kept,
                "last_graph": {
                    **last_graph_payload,
                    "labels": last_labels,
                },
            },
            f,
        )
    print(f"[OK  ] saved: {scores_path}")
    return True


# ============================================================================
# Stage 4: explain
# ============================================================================


def run_explain(
    *,
    dataset_dir: str,
    dataset: str,
    device: torch.device,
    device_index: int,
    checkpoint_dir: str,
    results_dir: str,
    explain_top_k: int,
    explain_k_hop: int,
    anomaly_threshold: float,
    epochs_for_shape: int,
    export_all_formats: bool = False,
    dpi: int = 1200,
    connected: bool = False,
) -> bool:
    _print_header("Stage 4: explain (minimal explanation subgraphs)")

    scores_path = os.path.join(results_dir, "anomaly_scores.pkl")
    if not os.path.exists(scores_path):
        print(f"[ERR ] missing `{scores_path}`; run `--stage eval` first")
        return False

    with open(scores_path, "rb") as f:
        saved = pkl.load(f)
    thr = float(saved.get("threshold", 0.5))
    last_graph = saved.get("last_graph", {}) if isinstance(saved, dict) else {}

    from utils.loaddata import (
        load_cic_metadata,
        load_entity_level_dataset,
        load_entity_level_dataset_with_cic,
        load_metadata_with_cic,
    )
    from utils.utils import set_random_seed
    from model.autoencoder import build_model
    from model.fusion import AnomalyScorer
    from model.contrastive import NodeLevelContrastive
    from model.explanation import prepare_explanation_builder_from_modules, SubgraphVisualizer

    # Keep explanation deterministic (contrastive projector is randomly initialized).
    set_random_seed(0)

    metadata = load_metadata_with_cic(dataset_dir)
    model_args = _build_default_model_args()
    model_args, use_cic, add_cic = _configure_model_args(
        base_args=model_args, dataset=dataset, epochs=epochs_for_shape, device_index=device_index, metadata=metadata
    )
    model = build_model(model_args).to(device)

    compat_path = os.path.join(checkpoint_dir, f"checkpoint-{dataset}.pt")
    final_path = os.path.join(checkpoint_dir, f"model_{dataset}_final.pt")
    loaded_ckpt = None
    
    if os.path.exists(compat_path):
        model.load_state_dict(torch.load(compat_path, map_location=device))
        print(f"[OK  ] loaded model: {compat_path}")
        loaded_ckpt = compat_path
    elif os.path.exists(final_path):
        ckpt = torch.load(final_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[OK  ] loaded model: {final_path}")
        loaded_ckpt = final_path
    else:
        # Try seed-specific checkpoints (s0, s1, ...)
        for seed_idx in range(10):
            seed_path = os.path.join(checkpoint_dir, f"checkpoint-{dataset}_s{seed_idx}.pt")
            if os.path.exists(seed_path):
                model.load_state_dict(torch.load(seed_path, map_location=device))
                print(f"[OK  ] loaded model: {seed_path}")
                loaded_ckpt = seed_path
                break
    
    if loaded_ckpt is None:
        print(f"[ERR ] missing checkpoint: `{compat_path}` or `{final_path}` or `checkpoint-{dataset}_s*.pt`")
        return False
    model.eval()

    n_test = int(metadata.get("n_test", 0))
    if n_test <= 0:
        print("[ERR ] Invalid metadata: n_test <= 0")
        return False

    loader = load_entity_level_dataset_with_cic if use_cic else load_entity_level_dataset
    g = loader(dataset_dir, "test", n_test - 1).to(device)
    if add_cic and "cic_scores" in g.ndata and "attr" in g.ndata:
        cic = g.ndata["cic_scores"].to(device=g.ndata["attr"].device, dtype=g.ndata["attr"].dtype)
        g.ndata["attr"] = torch.cat([g.ndata["attr"], cic], dim=-1)

    cic_meta = load_cic_metadata(dataset_dir)
    
    # Build proper node_id -> readable_name mapping
    def _build_node_names_map(cic_meta, dataset_dir):
        """Build node_id (int) -> readable_name (str) mapping"""
        names_data = cic_meta.get("names")
        entities_data = cic_meta.get("entities")
        
        # Try to get id_nodename_map from names.pkl
        uuid_to_name = {}
        if isinstance(names_data, dict):
            if "id_nodename_map" in names_data:
                uuid_to_name = names_data.get("id_nodename_map", {})
            else:
                # Maybe the dict itself is UUID -> name
                uuid_to_name = names_data
        
        # Build node_id -> uuid mapping from entities
        nodeid_to_uuid = {}
        if isinstance(entities_data, dict):
            # Check different possible structures
            if "node_id_to_uuid" in entities_data:
                nodeid_to_uuid = entities_data["node_id_to_uuid"]
            elif "uuid_to_node_id" in entities_data:
                for uuid, nid in entities_data["uuid_to_node_id"].items():
                    nodeid_to_uuid[nid] = uuid
            else:
                # Try: entities as {uuid: node_id} or {node_id: uuid}
                for k, v in entities_data.items():
                    if isinstance(k, int) and isinstance(v, str):
                        nodeid_to_uuid[k] = v
                    elif isinstance(k, str) and isinstance(v, int):
                        nodeid_to_uuid[v] = k
        
        # Combine: node_id -> uuid -> name
        result = {}
        for nid, uuid in nodeid_to_uuid.items():
            if uuid in uuid_to_name:
                result[nid] = uuid_to_name[uuid]
        
        print(f"[INFO] Built names_map: {len(result)} entries")
        return result if result else None
    
    names_map = _build_node_names_map(cic_meta, dataset_dir)
    types_map = cic_meta.get("types")

    node_contrast = NodeLevelContrastive(hidden_dim=int(getattr(model, "output_hidden_dim", 64))).to(device)
    anomaly_scorer = AnomalyScorer(n_sources=3).to(device)

    builder = prepare_explanation_builder_from_modules(
        g,
        model=model,
        cic_scores=g.ndata.get("cic_scores"),
        anomaly_scorer=anomaly_scorer,
        node_contrastive=node_contrast,
        names_map=names_map,
        types_map=types_map,
        anomaly_threshold=float(anomaly_threshold),
    )

    image_format = "jpeg"
    top = builder.get_top_anomaly_nodes(k=int(explain_top_k))
    if not top:
        print("[WARN] no anomaly nodes found")
        return True

    # Output figures to {data_root}/figures/{dataset}/ (parallel to data directory)
    # e.g., /hy-tmp/code/figures/clear/
    data_root = os.path.dirname(dataset_dir)  # /hy-tmp/code/data -> /hy-tmp/code
    figures_dir = os.path.join(data_root, "figures", dataset)
    os.makedirs(figures_dir, exist_ok=True)
    print(f"[INFO] figures will be saved to: {figures_dir}")
    
    # ===== 连通子图模式 =====
    if connected:
        print(f"[INFO] 使用连通最小解释子图模式 (top_k={explain_top_k})")
        from model.connected_explanation import ConnectedExplanationBuilder, visualize_connected_subgraph
        
        # 获取异常分数
        with torch.no_grad():
            emb = model.embed(g)
            if isinstance(emb, tuple):
                emb = emb[0]
            recon = model.node_reconstruction_error(g) if hasattr(model, "node_reconstruction_error") else torch.zeros(g.num_nodes(), device=device)
            cic_data = g.ndata.get("cic_scores", torch.zeros((g.num_nodes(), 4), device=device))
            contrast = node_contrast.anomaly_score(emb, cic_data, threshold=float(anomaly_threshold))
            fused = anomaly_scorer.compute_anomaly_score(cic_data, contrastive_score=contrast, recon_error=recon)
            fused = torch.nan_to_num(fused, nan=0.0).clamp(0.0, 1.0)
        
        connected_builder = ConnectedExplanationBuilder(
            graph=g,
            anomaly_scores=fused,
            cic_scores=cic_data,
            names_map=names_map,
            types_map=types_map,
            alpha=0.5,
            bridge_budget=15,
        )
        
        subgraph_data, layers = connected_builder.build(
            top_k=int(explain_top_k),
            anomaly_threshold=float(anomaly_threshold),
        )
        
        # 可视化三层连通子图
        output_path = os.path.join(figures_dir, "connected_explanation")
        try:
            visualize_connected_subgraph(
                subgraph_data, layers, output_path,
                format="jpeg", dpi=dpi,
            )
        except Exception as e:
            print(f"[WARN] visualize_connected_subgraph failed: {e}")
        
        # 保存数据
        out_pkl = os.path.join(results_dir, "connected_subgraph.pkl")
        with open(out_pkl, "wb") as f:
            pkl.dump({"subgraph": subgraph_data, "layers": layers}, f)
        print(f"[OK  ] saved: {out_pkl}")
        return True
    
    # ===== 独立子图模式 (默认) =====
    visualizer = SubgraphVisualizer(output_dir=figures_dir)

    print(f"[INFO] threshold={thr:.4f} top_k={explain_top_k} k_hop={explain_k_hop}")
    print("[INFO] top anomalies (first 10):")
    for rank, (nid, score) in enumerate(top[: min(10, len(top))], start=1):
        print(f"  {rank:2d}. node={int(nid)} score={float(score):.6f}")

    subgraphs = []
    for nid, _ in top[: int(explain_top_k)]:
        subg = builder.build_subgraph(center_node=int(nid), k_hop=int(explain_k_hop), threshold=0.0)
        builder.find_attack_path(subg)
        try:
            visualizer.visualize_subgraph(
                subg,
                filename=f"subgraph_node_{int(nid)}",
                format=image_format,
                show_scores=False,
                highlight_attack_path=True,
                export_all_formats=export_all_formats,
                dpi=dpi,
            )
        except Exception as e:
            print(f"[WARN] visualize_subgraph failed for node={int(nid)}: {e}")
        subgraphs.append(subg)

    try:
        visualizer.visualize_attack_summary(
            subgraphs,
            filename="attack_summary",
            format=image_format,
            export_all_formats=export_all_formats,
            dpi=dpi,
        )
    except Exception as e:
        print(f"[WARN] visualize_attack_summary failed: {e}")

    out_pkl = os.path.join(results_dir, "subgraphs_data.pkl")
    with open(out_pkl, "wb") as f:
        pkl.dump(subgraphs, f)
    print(f"[OK  ] saved: {out_pkl}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="MAGIC end-to-end pipeline")
    parser.add_argument("--stage", required=True, choices=["all", "preprocess", "train", "eval", "explain"])
    parser.add_argument("--data_dir", required=True, help="Dataset dir or dataset root (will append /{dataset})")
    parser.add_argument("--dataset", default="theia", choices=["theia", "cadets", "clear", "trace"])

    parser.add_argument("--device", default="cuda:0", help="cpu | cuda:0 | 0 | -1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=50)

    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--results_dir", default="./results")

    parser.add_argument("--explain_top_k", type=int, default=10)
    parser.add_argument("--explain_k_hop", type=int, default=2)
    
    # Connected subgraph mode (default on)
    parser.add_argument("--connected", action="store_true", default=True,
                        help="Use connected minimal explanation subgraph (research design)")
    parser.add_argument("--no-connected", dest="connected", action="store_false",
                        help="Disable connected minimal explanation subgraph")
    
    # 论文级别图表导出选项
    parser.add_argument("--all", action="store_true", default=False,
                        help="Export all paper formats: PDF, EPS, SVG, PNG, TIFF, JPEG (default: JPEG only)")
    parser.add_argument("--export_all_formats", action="store_true", default=False,
                        help="(deprecated) Same as --all")
    parser.add_argument("--dpi", type=int, default=1200,
                        help="Image DPI for paper-quality figures (default: 1200)")

    # threshold to split "normal" prototype in contrastive scoring
    parser.add_argument("--cic_anomaly_threshold", type=float, default=0.5)

    args = parser.parse_args()
    args.export_all_formats = bool(args.all or args.export_all_formats)
    
    # 连通模式下默认 top_k=5
    if args.connected and args.explain_top_k == 10:
        args.explain_top_k = 5

    args.checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_dir, args.dataset)
    # For eval/explain stages, use results/explain/{dataset}/
    if args.stage in ("eval", "explain", "all"):
        args.results_dir = os.path.join(args.results_dir, "explain", args.dataset)
    else:
        args.results_dir = _ensure_dataset_subdir(args.results_dir, args.dataset)
    os.makedirs(args.results_dir, exist_ok=True)

    dataset_dir, _ = resolve_dataset_dir(args.data_dir, args.dataset)
    device, device_index = _device_from_string(args.device)

    print(f"[INFO] stage={args.stage} dataset={args.dataset}")
    print(f"[INFO] dataset_dir={dataset_dir}")
    print(f"[INFO] device={device}")

    stages = ["preprocess", "train", "eval", "explain"] if args.stage == "all" else [args.stage]
    for s in stages:
        if s == "preprocess":
            ok = run_preprocess(dataset_dir, args.dataset)
        elif s == "train":
            ok = run_train(
                dataset_dir=dataset_dir,
                dataset=args.dataset,
                device=device,
                device_index=device_index,
                epochs=args.epochs,
                checkpoint_dir=args.checkpoint_dir,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                seed=args.seed,
            )
        elif s == "eval":
            ok = run_eval(
                dataset_dir=dataset_dir,
                dataset=args.dataset,
                device=device,
                device_index=device_index,
                checkpoint_dir=args.checkpoint_dir,
                results_dir=args.results_dir,
                anomaly_threshold=args.cic_anomaly_threshold,
                epochs_for_shape=args.epochs,
            )
        elif s == "explain":
            ok = run_explain(
                dataset_dir=dataset_dir,
                dataset=args.dataset,
                device=device,
                device_index=device_index,
                checkpoint_dir=args.checkpoint_dir,
                results_dir=args.results_dir,
                explain_top_k=args.explain_top_k,
                explain_k_hop=args.explain_k_hop,
                anomaly_threshold=args.cic_anomaly_threshold,
                epochs_for_shape=args.epochs,
                export_all_formats=args.export_all_formats,
                dpi=args.dpi,
                connected=args.connected,
            )
        else:
            ok = False

        if not ok:
            raise SystemExit(1)

    _print_header("Pipeline finished")


if __name__ == "__main__":
    main()
