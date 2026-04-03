"""
Evaluation utilities (KNN metrics, caching) used by eval.py.
"""

import os
import json
import random
import time
import pickle as pkl
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from utils.utils import set_random_seed
from utils.loaddata import transform_graph, load_batch_level_dataset

# ============================================================================
# GPU-accelerated KNN using faiss (with sklearn fallback)
# ============================================================================

_FAISS_GPU_AVAILABLE = False
_faiss = None

try:
    import faiss
    _faiss = faiss
    if faiss.get_num_gpus() > 0:
        _FAISS_GPU_AVAILABLE = True
        print("[EVAL] faiss-gpu available, using GPU-accelerated KNN")
    else:
        print("[EVAL] faiss available but no GPU, using CPU faiss")
except ImportError:
    print("[EVAL] faiss not installed, falling back to sklearn KNN (slow)")


class FastKNN:
    """GPU-accelerated KNN using faiss, with sklearn fallback."""
    
    def __init__(self, n_neighbors: int = 10, use_gpu: bool = True):
        self.n_neighbors = n_neighbors
        self.use_gpu = use_gpu and _FAISS_GPU_AVAILABLE
        self.index = None
        self._sklearn_nbrs = None
        self._faiss_res = None
    
    def fit(self, x_train: np.ndarray):
        """Build KNN index from training data."""
        x_train = np.ascontiguousarray(x_train.astype(np.float32))
        d = x_train.shape[1]
        
        if _faiss is not None:
            # Use faiss
            self.index = _faiss.IndexFlatL2(d)
            if self.use_gpu:
                if self._faiss_res is None:
                    self._faiss_res = _faiss.StandardGpuResources()
                self.index = _faiss.index_cpu_to_gpu(self._faiss_res, 0, self.index)
            self.index.add(x_train)
        else:
            # Fallback to sklearn
            self._sklearn_nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1)
            self._sklearn_nbrs.fit(x_train)
        
        return self
    
    def kneighbors(self, x_query: np.ndarray, n_neighbors: int = None, *, batch_size: int = None, show_progress: bool = None):
        """Find k nearest neighbors for query points (optionally batched)."""
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        x_query = np.ascontiguousarray(x_query.astype(np.float32))
        total = x_query.shape[0]
        if batch_size is None:
            batch_size = 50000 if total >= 100000 else total
        if show_progress is None:
            show_progress = total > batch_size

        if self.index is not None:
            # faiss returns squared L2 distances
            if total <= batch_size:
                distances_sq, indices = self.index.search(x_query, n_neighbors)
                distances = np.sqrt(np.maximum(distances_sq, 0))  # sqrt and clamp negatives
                return distances, indices
            distances_list = []
            indices_list = []
            iterator = range(0, total, batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="KNN search", leave=False)
            for start in iterator:
                end = min(start + batch_size, total)
                distances_sq, indices = self.index.search(x_query[start:end], n_neighbors)
                distances = np.sqrt(np.maximum(distances_sq, 0))
                distances_list.append(distances)
                indices_list.append(indices)
            return np.concatenate(distances_list, axis=0), np.concatenate(indices_list, axis=0)
        else:
            # sklearn fallback
            if total <= batch_size:
                return self._sklearn_nbrs.kneighbors(x_query, n_neighbors=n_neighbors)
            distances_list = []
            indices_list = []
            iterator = range(0, total, batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="KNN search", leave=False)
            for start in iterator:
                end = min(start + batch_size, total)
                distances, indices = self._sklearn_nbrs.kneighbors(
                    x_query[start:end], n_neighbors=n_neighbors
                )
                distances_list.append(distances)
                indices_list.append(indices)
            return np.concatenate(distances_list, axis=0), np.concatenate(indices_list, axis=0)




def batch_level_evaluation(
    model,
    pooler,
    device,
    method,
    dataset,
    n_dim=0,
    e_dim=0,
    *,
    log_path=None,
    log_extra=None,
    show_progress=False,
    knn_repeat=100,
    verbose=True,
):
    model.eval()
    x_list = []
    y_list = []
    data = load_batch_level_dataset(dataset)
    full = data['full_index']
    graphs = data['dataset']
    with torch.no_grad():
        iterator = full
        if show_progress and len(full) > 1:
            iterator = tqdm(full, desc="Batch eval", leave=False)
        for i in iterator:
            g = transform_graph(graphs[i][0], n_dim, e_dim).to(device)
            label = graphs[i][1]
            out = model.embed(g)
            if dataset != 'wget':
                out = pooler(g, out).cpu().numpy()
            else:
                out = pooler(g, out, n_types=data['n_feat']).cpu().numpy()
            y_list.append(label)
            x_list.append(out)
    x = np.concatenate(x_list, axis=0)
    y = np.array(y_list)
    if 'knn' in method:
        test_auc, test_std = evaluate_batch_level_using_knn(
            knn_repeat,
            dataset,
            x,
            y,
            log_path=log_path,
            log_extra=log_extra,
            verbose=verbose,
        )
    else:
        raise NotImplementedError
    return test_auc, test_std


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _compute_tpr_at_fpr(y_true, scores, target_fpr):
    """计算在指定 FPR 水平下的 TPR (True Positive Rate)"""
    if np.unique(y_true).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    # 找到最接近 target_fpr 的点
    idx = np.searchsorted(fpr, target_fpr)
    if idx >= len(fpr):
        return float(tpr[-1])
    if idx == 0:
        return float(tpr[0])
    # 线性插值
    if fpr[idx] == target_fpr:
        return float(tpr[idx])
    ratio = (target_fpr - fpr[idx - 1]) / (fpr[idx] - fpr[idx - 1] + 1e-9)
    return float(tpr[idx - 1] + ratio * (tpr[idx] - tpr[idx - 1]))


def _compute_fpr_at_tpr(y_true, scores, target_tpr):
    """计算在指定 TPR 水平下的 FPR (False Positive Rate)"""
    if np.unique(y_true).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    # 找到最接近 target_tpr 的点（注意 tpr 是单调递增的）
    idx = np.searchsorted(tpr, target_tpr)
    if idx >= len(tpr):
        return float(fpr[-1])
    if idx == 0:
        return float(fpr[0])
    # 线性插值
    if tpr[idx] == target_tpr:
        return float(fpr[idx])
    ratio = (target_tpr - tpr[idx - 1]) / (tpr[idx] - tpr[idx - 1] + 1e-9)
    return float(fpr[idx - 1] + ratio * (fpr[idx] - fpr[idx - 1]))


def evaluate_batch_level_using_knn(
    repeat,
    dataset,
    embeddings,
    labels,
    *,
    log_path=None,
    log_extra=None,
    verbose=True,
):
    x, y = embeddings, labels
    if dataset == 'streamspot':
        train_count = 400
    else:
        train_count = 100
    n_neighbors = max(1, min(int(train_count * 0.02), 10))
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    if repeat != -1:
        prec_list = []
        rec_list = []
        f1_list = []
        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []
        auc_list = []
        for s in range(repeat):
            set_random_seed(s)
            np.random.shuffle(benign_idx)
            np.random.shuffle(attack_idx)
            x_train = x[benign_idx[:train_count]]
            x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
            y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
            x_train_mean = x_train.mean(axis=0)
            x_train_std = x_train.std(axis=0)
            x_train = (x_train - x_train_mean) / (x_train_std + 1e-6)
            x_test = (x_test - x_train_mean) / (x_train_std + 1e-6)

            nbrs = FastKNN(n_neighbors=n_neighbors)
            nbrs.fit(x_train)
            distances, _ = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
            mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
            distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

            score = distances.mean(axis=1) / mean_distance

            auc = roc_auc_score(y_test, score)
            prec, rec, threshold = precision_recall_curve(y_test, score)
            f1 = 2 * prec * rec / (rec + prec + 1e-9)
            max_f1_idx = np.argmax(f1)
            best_thres = threshold[max_f1_idx]
            prec_list.append(prec[max_f1_idx])
            rec_list.append(rec[max_f1_idx])
            f1_list.append(f1[max_f1_idx])

            tn = 0
            fn = 0
            tp = 0
            fp = 0
            for i in range(len(y_test)):
                if y_test[i] == 1.0 and score[i] >= best_thres:
                    tp += 1
                if y_test[i] == 1.0 and score[i] < best_thres:
                    fn += 1
                if y_test[i] == 0.0 and score[i] < best_thres:
                    tn += 1
                if y_test[i] == 0.0 and score[i] >= best_thres:
                    fp += 1
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            tn_list.append(tn)
            auc_list.append(auc)

        metrics = {
            "auc_mean": float(np.mean(auc_list)),
            "auc_std": float(np.std(auc_list)),
            "f1_mean": float(np.mean(f1_list)),
            "f1_std": float(np.std(f1_list)),
            "precision_mean": float(np.mean(prec_list)),
            "precision_std": float(np.std(prec_list)),
            "recall_mean": float(np.mean(rec_list)),
            "recall_std": float(np.std(rec_list)),
            "tn_mean": float(np.mean(tn_list)),
            "fn_mean": float(np.mean(fn_list)),
            "tp_mean": float(np.mean(tp_list)),
            "fp_mean": float(np.mean(fp_list)),
        }
        if log_path:
            payload = dict(log_extra or {})
            payload.update(metrics)
            _write_json(log_path, payload)
        if verbose:
            print('AUC: {}+{}'.format(metrics["auc_mean"], metrics["auc_std"]))
            print('F1: {}+{}'.format(metrics["f1_mean"], metrics["f1_std"]))
            print('PRECISION: {}+{}'.format(metrics["precision_mean"], metrics["precision_std"]))
            print('RECALL: {}+{}'.format(metrics["recall_mean"], metrics["recall_std"]))
            print('TN: {}+{}'.format(metrics["tn_mean"], 0.0))
            print('FN: {}+{}'.format(metrics["fn_mean"], 0.0))
            print('TP: {}+{}'.format(metrics["tp_mean"], 0.0))
            print('FP: {}+{}'.format(metrics["fp_mean"], 0.0))
        return np.mean(auc_list), np.std(auc_list)
    else:
        set_random_seed(0)
        np.random.shuffle(benign_idx)
        np.random.shuffle(attack_idx)
        x_train = x[benign_idx[:train_count]]
        x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
        y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
        x_train_mean = x_train.mean(axis=0)
        x_train_std = x_train.std(axis=0)
        x_train = (x_train - x_train_mean) / (x_train_std + 1e-6)
        x_test = (x_test - x_train_mean) / (x_train_std + 1e-6)

        nbrs = FastKNN(n_neighbors=n_neighbors)
        nbrs.fit(x_train)
        distances, _ = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
        mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

        score = distances.mean(axis=1) / mean_distance
        auc = roc_auc_score(y_test, score)
        prec, rec, threshold = precision_recall_curve(y_test, score)
        f1 = 2 * prec * rec / (rec + prec + 1e-9)
        best_idx = np.argmax(f1)
        best_thres = threshold[best_idx]

        tn = 0
        fn = 0
        tp = 0
        fp = 0
        for i in range(len(y_test)):
            if y_test[i] == 1.0 and score[i] >= best_thres:
                tp += 1
            if y_test[i] == 1.0 and score[i] < best_thres:
                fn += 1
            if y_test[i] == 0.0 and score[i] < best_thres:
                tn += 1
            if y_test[i] == 0.0 and score[i] >= best_thres:
                fp += 1
        metrics = {
            "auc": float(auc),
            "f1": float(f1[best_idx]),
            "precision": float(prec[best_idx]),
            "recall": float(rec[best_idx]),
            "tn": float(tn),
            "fn": float(fn),
            "tp": float(tp),
            "fp": float(fp),
        }
        if log_path:
            payload = dict(log_extra or {})
            payload.update(metrics)
            _write_json(log_path, payload)
        if verbose:
            print('AUC: {}'.format(metrics["auc"]))
            print('F1: {}'.format(metrics["f1"]))
            print('PRECISION: {}'.format(metrics["precision"]))
            print('RECALL: {}'.format(metrics["recall"]))
            print('TN: {}'.format(metrics["tn"]))
            print('FN: {}'.format(metrics["fn"]))
            print('TP: {}'.format(metrics["tp"]))
            print('FP: {}'.format(metrics["fp"]))
        return auc, 0.0


def evaluate_entity_level_using_knn(
    dataset,
    x_train,
    x_test,
    y_test,
    *,
    log_path=None,
    log_extra=None,
    verbose=True,
):
    if np.unique(y_test).size < 2:
        metrics = {
            "auc": float("nan"),
            "pr_auc": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "tpr_at_fpr_0.001": float("nan"),
            "tpr_at_fpr_0.01": float("nan"),
            "fpr_at_tpr_0.95": float("nan"),
            "fpr_at_tpr_0.99": float("nan"),
            "tn": float("nan"),
            "fn": float("nan"),
            "tp": float("nan"),
            "fp": float("nan"),
            "threshold": float("nan"),
        }
        if log_path:
            payload = dict(log_extra or {})
            payload.update(metrics)
            _write_json(log_path, payload)
        if verbose:
            print("[WARN] y_test has a single class; metrics are undefined.")
        return float("nan"), 0.0, metrics, None
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / (x_train_std + 1e-6)
    x_test = (x_test - x_train_mean) / (x_train_std + 1e-6)

    if dataset == 'cadets':
        n_neighbors = 200
    else:
        n_neighbors = 10

    nbrs = FastKNN(n_neighbors=n_neighbors)
    nbrs.fit(x_train)

    idx = list(range(x_train.shape[0]))
    random.shuffle(idx)
    distances, _ = nbrs.kneighbors(x_train[idx][:min(50000, x_train.shape[0])], n_neighbors=n_neighbors)
    mean_distance = distances.mean()
    distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
    distances = distances.mean(axis=1)
    score = distances / mean_distance
    del distances
    auc = roc_auc_score(y_test, score)
    prec, rec, threshold = precision_recall_curve(y_test, score)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    best_idx = int(np.argmax(f1)) if f1.size else -1
    # Use the threshold that maximizes F1 score (universal approach)
    if best_idx < 0:
        best_idx = 0
    if threshold.size == 0:
        best_thres = 0.0
    else:
        if best_idx >= len(threshold):
            best_idx = len(threshold) - 1
        best_thres = threshold[best_idx]

    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and score[i] >= best_thres:
            tp += 1
        if y_test[i] == 1.0 and score[i] < best_thres:
            fn += 1
        if y_test[i] == 0.0 and score[i] < best_thres:
            tn += 1
        if y_test[i] == 0.0 and score[i] >= best_thres:
            fp += 1

    # 计算额外指标
    pr_auc = average_precision_score(y_test, score)
    tpr_at_fpr_001 = _compute_tpr_at_fpr(y_test, score, 0.001)  # TPR @ FPR=0.1%
    tpr_at_fpr_01 = _compute_tpr_at_fpr(y_test, score, 0.01)    # TPR @ FPR=1%
    fpr_at_tpr_95 = _compute_fpr_at_tpr(y_test, score, 0.95)    # FPR @ TPR=95%
    fpr_at_tpr_99 = _compute_fpr_at_tpr(y_test, score, 0.99)    # FPR @ TPR=99%

    metrics = {
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1[best_idx]) if f1.size else float("nan"),
        "precision": float(prec[best_idx]) if prec.size else float("nan"),
        "recall": float(rec[best_idx]) if rec.size else float("nan"),
        "tpr_at_fpr_0.001": float(tpr_at_fpr_001),
        "tpr_at_fpr_0.01": float(tpr_at_fpr_01),
        "fpr_at_tpr_0.95": float(fpr_at_tpr_95),
        "fpr_at_tpr_0.99": float(fpr_at_tpr_99),
        "tn": float(tn),
        "fn": float(fn),
        "tp": float(tp),
        "fp": float(fp),
        "threshold": float(best_thres),
    }
    if log_path:
        payload = dict(log_extra or {})
        payload.update(metrics)
        _write_json(log_path, payload)
    if verbose:
        print('AUC: {:.6f}'.format(metrics["auc"]))
        print('PR-AUC: {:.6f}'.format(metrics["pr_auc"]))
        print('F1: {:.6f}'.format(metrics["f1"]))
        print('PRECISION: {:.6f}'.format(metrics["precision"]))
        print('RECALL: {:.6f}'.format(metrics["recall"]))
        print('TPR@FPR=0.1%: {:.6f}'.format(metrics["tpr_at_fpr_0.001"]))
        print('TPR@FPR=1%: {:.6f}'.format(metrics["tpr_at_fpr_0.01"]))
        print('FPR@TPR=95%: {:.6f}'.format(metrics["fpr_at_tpr_0.95"]))
        print('FPR@TPR=99%: {:.6f}'.format(metrics["fpr_at_tpr_0.99"]))
        print('TP: {} | FP: {} | TN: {} | FN: {}'.format(int(tp), int(fp), int(tn), int(fn)))
    return auc, 0.0, metrics, None
