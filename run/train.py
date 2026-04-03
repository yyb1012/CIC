"""
Train entrypoint for MAGIC.

- Requires CUDA (GPU) to run.
- Saves model checkpoints and training metrics to disk.
"""

import os
import sys
import json
import random
import time
import torch
import warnings
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
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from model.train import batch_level_train
from utils.utils import set_random_seed, create_optimizer
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


def _write_json(path: str, payload: dict, comment: str = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if comment:
            payload = dict(payload)
            payload.setdefault("_comment", comment)
        json.dump(payload, f, indent=2, ensure_ascii=True)


def extract_dataloaders(entries, batch_size):
    random.shuffle(entries)
    train_idx = torch.arange(len(entries))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(entries, batch_size=batch_size, sampler=train_sampler)
    return train_loader


def train_single_seed(main_args, device, dataset_name, seed_value: int):
    """训练单个 seed 的模型"""
    set_random_seed(seed_value)
    
    # 设置默认参数
    if dataset_name == 'streamspot':
        main_args.num_hidden = 256
        main_args.num_layers = 4
        default_epoch = 5
    elif dataset_name == 'wget':
        main_args.num_hidden = 256
        main_args.num_layers = 4
        default_epoch = 2
    else:
        main_args.num_hidden = 64
        main_args.num_layers = 3
        default_epoch = 200
    if int(getattr(main_args, "max_epoch", 0)) <= 0:
        main_args.max_epoch = default_epoch

    if dataset_name == 'streamspot' or dataset_name == 'wget':
        if dataset_name == 'streamspot':
            default_batch_size = 12
        else:
            default_batch_size = 1
        batch_size = int(getattr(main_args, "batch_size", 0))
        if batch_size <= 0:
            batch_size = default_batch_size
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        graphs = dataset['dataset']
        train_index = dataset['train_index']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        model = build_model(main_args)
        model = model.to(device)
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)
        train_start = time.perf_counter()
        model = batch_level_train(
            model,
            graphs,
            (extract_dataloaders(train_index, batch_size)),
            optimizer,
            main_args.max_epoch,
            device,
            main_args.n_dim,
            main_args.e_dim,
        )
        train_time = time.perf_counter() - train_start
        ckpt_path = _checkpoint_path(dataset_name, seed_value)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        _write_json(
            os.path.join(_dataset_dir(DATA_COMPASS_DIR, dataset_name), f"train_metrics_{dataset_name}_s{seed_value}.json"),
            {
                "dataset": dataset_name,
                "device": str(device),
                "seed": seed_value,
                "epochs": main_args.max_epoch,
                "train_time_seconds": float(train_time),
                "note": "batch-level training; per-epoch loss is not recorded.",
            },
            comment="Training metrics for batch-level dataset; per-epoch loss is not recorded.",
        )
    else:
        metadata = load_metadata_with_cic(dataset_name)
        use_cic = bool(getattr(main_args, "use_cic", False)) and bool(metadata.get("has_cic_scores", False))
        add_cic_to_features = use_cic and bool(getattr(main_args, "cic_as_node_feature", False))

        main_args.n_dim = metadata['node_feature_dim'] + (4 if add_cic_to_features else 0)
        main_args.e_dim = metadata['edge_feature_dim']
        model = build_model(main_args)
        model = model.to(device)
        model.train()
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)
        n_train = metadata['n_train']
        all_indices = list(range(n_train))
        random.shuffle(all_indices)
        val_ratio = float(getattr(main_args, "val_ratio", 0.0))
        val_count = 0
        if val_ratio > 0 and n_train > 1:
            val_count = max(1, int(round(n_train * val_ratio)))
            val_count = min(val_count, n_train - 1)
        val_indices = set(all_indices[:val_count])
        train_indices = [i for i in all_indices if i not in val_indices]
        use_early_stop = bool(getattr(main_args, "early_stop", False)) and val_count > 0
        loader = load_entity_level_dataset_with_cic if use_cic else load_entity_level_dataset
        patience = int(getattr(main_args, "early_stop_patience", 5))
        min_delta = float(getattr(main_args, "early_stop_min_delta", 1e-4))
        val_every_n = max(1, int(getattr(main_args, "val_every_n", 1)))
        best_val = float("inf")
        best_state = None
        stale_epochs = 0
        train_history = []
        print(f"[TRAIN] dataset={dataset_name}, train={len(train_indices)} graphs, val={len(val_indices)} graphs")
        print(f"[TRAIN] max_epoch={main_args.max_epoch}, early_stop={use_early_stop}, val_every_n={val_every_n}, use_cic={use_cic}")
        
        
        train_start = time.perf_counter()
        def run_epoch(indices, train_mode: bool, desc: str = "") -> float:
            total_loss = 0.0
            denom = max(1, len(indices))
            if train_mode:
                model.train()
            else:
                model.eval()
            pbar = tqdm(indices, desc=desc, leave=False) if len(indices) > 1 else indices
            for i in pbar:
                g = loader(dataset_name, 'train', i).to(device)
                if add_cic_to_features and 'cic_scores' in g.ndata and 'attr' in g.ndata:
                    cic = g.ndata['cic_scores'].to(device=g.ndata['attr'].device, dtype=g.ndata['attr'].dtype)
                    g.ndata['attr'] = torch.cat([g.ndata['attr'], cic], dim=-1)
                if train_mode:
                    loss = model(g)
                    loss = loss / denom
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        loss = model(g)
                        loss = loss / denom
                total_loss += loss.item()
                del g
            return total_loss

        for epoch in range(main_args.max_epoch):
            stop_training = False
            epoch_loss = run_epoch(train_indices, train_mode=True, desc=f"Epoch {epoch+1}/{main_args.max_epoch}")
            entry = {"epoch": int(epoch), "loss": float(epoch_loss)}
            
            # 打印训练进度
            progress_msg = f"[Epoch {epoch+1:3d}/{main_args.max_epoch}] train_loss={epoch_loss:.6f}"
            
            if use_early_stop and (epoch % val_every_n == 0):
                val_loss = run_epoch(sorted(val_indices), train_mode=False, desc="Validation")
                entry["val_loss"] = float(val_loss)
                progress_msg += f" | val_loss={val_loss:.6f}"
                
                if val_loss + min_delta < best_val:
                    best_val = val_loss
                    best_state = {
                        k: v.detach().cpu().clone() if torch.is_tensor(v) else v
                        for k, v in model.state_dict().items()
                    }
                    stale_epochs = 0
                    progress_msg += " [best]"
                else:
                    stale_epochs += 1
                    progress_msg += f" (stale: {stale_epochs}/{patience})"
                    if stale_epochs >= patience:
                        progress_msg += " [early_stop]"
                        stop_training = True
            
            print(progress_msg)
            train_history.append(entry)
            if stop_training:
                print(f"[EARLY STOP] Stopped at epoch {epoch+1}, best_val_loss={best_val:.6f}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        train_time = time.perf_counter() - train_start
        ckpt_path = _checkpoint_path(dataset_name, seed_value)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        _write_json(
            os.path.join(_dataset_dir(DATA_COMPASS_DIR, dataset_name), f"train_metrics_{dataset_name}_s{seed_value}.json"),
            {
                "dataset": dataset_name,
                "device": str(device),
                "seed": seed_value,
                "epochs": main_args.max_epoch,
                "train_time_seconds": float(train_time),
                "history": train_history,
            },
            comment="Training metrics for entity-level dataset; history includes per-epoch loss and optional val_loss.",
        )
    print(f"[TRAIN] seed={seed_value} done")
    return


def main(main_args):
    """多 seed 训练入口"""
    device = _require_cuda(int(main_args.device))
    dataset_name = main_args.dataset
    num_seeds = max(1, int(getattr(main_args, "num_seeds", 5)))
    seed_start = int(getattr(main_args, "seed_start", 0))
    
    print(f"[TRAIN] dataset={dataset_name} num_seeds={num_seeds} seed_start={seed_start}")
    
    for idx in range(num_seeds):
        seed_value = seed_start + idx
        print(f"\n{'='*60}")
        print(f"[TRAIN] Starting seed {seed_value} ({idx+1}/{num_seeds})")
        print(f"{'='*60}")
        train_single_seed(main_args, device, dataset_name, seed_value)
    
    print(f"\n[TRAIN] All {num_seeds} seeds completed.")
    return


if __name__ == '__main__':
    args = build_args()
    main(args)
