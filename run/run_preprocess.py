"""
Step 1: 生成 CIC 增强数据

使用方法:
    python run_preprocess.py --dataset theia
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _resolve_dataset_dir(args) -> str:
    from utils.loaddata import resolve_data_dir

    if args.data_dir:
        base = resolve_data_dir(args.data_dir)
        if os.path.exists(os.path.join(base, "train.pkl")) or os.path.exists(os.path.join(base, "metadata_cic.json")):
            return base
        candidate = os.path.join(base, args.dataset)
        if os.path.exists(os.path.join(candidate, "train.pkl")) or os.path.exists(
            os.path.join(candidate, "metadata_cic.json")
        ):
            return candidate
        return base

    return resolve_data_dir(args.dataset)


def main():
    parser = argparse.ArgumentParser(description='生成CIC增强数据')
    parser.add_argument('--dataset', type=str, default='theia', choices=['theia', 'cadets', 'clear', 'trace'])
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录')
    parser.add_argument('--workers', type=int, default=1, help='并行处理进程数（仅在Linux fork下生效）')
    args = parser.parse_args()
    
    data_dir = _resolve_dataset_dir(args)
    if os.path.isdir(data_dir):
        os.environ.setdefault("DATA_ROOT", os.path.dirname(data_dir))
    
    print("=" * 60)
    print("生成 CIC 增强数据")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"数据目录: {data_dir}")
    
    # 检查必要文件
    required_files = ['train.pkl', 'test.pkl', 'entities.pkl', 'invariant_tracking.pkl']
    print("\n检查必要文件:")
    for f in required_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {f} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {f} (未找到)")
    
    # 生成 CIC 数据
    print("\n[RUN] 正在生成 CIC 增强数据...")
    
    from utils.loaddata import preload_entity_level_dataset_with_cic
    preload_entity_level_dataset_with_cic(data_dir, compute_cic=True, workers=max(1, args.workers))
    
    # 检查输出
    metadata_path = os.path.join(data_dir, 'metadata_cic.json')
    if os.path.exists(metadata_path):
        print(f"\n[OK] 已生成: {metadata_path}")
    else:
        print(f"\n[ERR] 生成失败")


if __name__ == '__main__':
    main()
