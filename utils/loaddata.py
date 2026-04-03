import pickle as pkl
import time
import torch.nn.functional as F
import dgl
import networkx as nx
import json
from tqdm import tqdm
import os
import gc
import multiprocessing as mp


def resolve_data_dir(path: str) -> str:
    """
    Resolve dataset directory when project and datasets are in different roots.

    - If `path` is absolute (or ~), use it directly.
    - Else try `$DATA_ROOT/{path}`.
    - Else fall back to `./data/{path}`.
    """
    if not path:
        return path

    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded

    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        return os.path.join(os.path.expanduser(data_root), path)

    candidate = os.path.join(".", "data", path)
    return candidate


def _resolve_feature_dims_from_metadata(data_path: str):
    metadata_path = os.path.join(data_path, "metadata.json")
    if not os.path.exists(metadata_path):
        return None, None, None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return None, None, None
    node_feature_dim = metadata.get("node_feature_dim")
    edge_feature_dim = metadata.get("edge_feature_dim")
    malicious = metadata.get("malicious")
    if isinstance(node_feature_dim, int) and isinstance(edge_feature_dim, int):
        return node_feature_dim, edge_feature_dim, malicious
    return None, None, malicious


def _resolve_feature_dims_from_type_mappings(data_path: str):
    type_mappings_path = os.path.join(data_path, "type_mappings.pkl")
    if not os.path.exists(type_mappings_path):
        return None, None
    try:
        with open(type_mappings_path, "rb") as f:
            mappings = pkl.load(f)
    except Exception:
        return None, None
    node_dict = mappings.get("node_type_dict") or {}
    edge_dict = mappings.get("edge_type_dict") or {}
    node_feature_dim = None
    edge_feature_dim = None
    if node_dict:
        node_feature_dim = max(int(v) for v in node_dict.values()) + 1
    if edge_dict:
        edge_feature_dim = max(int(v) for v in edge_dict.values()) + 1
    return node_feature_dim, edge_feature_dim


def _update_feature_dims_from_graphs(graphs, node_feature_dim: int, edge_feature_dim: int):
    for g in graphs:
        for n in g.get("nodes", []):
            if "type" in n:
                node_feature_dim = max(node_feature_dim, int(n["type"]))
        for e in g.get("links", []):
            if "type" in e:
                edge_feature_dim = max(edge_feature_dim, int(e["type"]))
    return node_feature_dim, edge_feature_dim


_CIC_WORKER_CTX = {}


def _node_link_to_nx(g_data):
    multigraph = bool(g_data.get('multigraph', False))
    directed = bool(g_data.get('directed', True))
    return nx.node_link_graph(g_data, directed=directed, multigraph=multigraph)


def _nx_node_uuids(g_nx):
    return [g_nx.nodes[n].get('uuid') for n in g_nx.nodes()]


def _get_mp_context():
    if os.name == "posix":
        try:
            return mp.get_context("fork")
        except ValueError:
            return mp.get_context()
    return mp.get_context()


def _init_cic_worker():
    # Context is inherited via fork.
    return None


def _process_cic_graph(args):
    idx, g_data, split_name, data_path = args
    out_path = os.path.join(data_path, f'{split_name}_cic{idx}.pkl')
    
    # 跳过已处理的文件
    if os.path.exists(out_path):
        return idx, True, None
    
    try:
        g_nx = _node_link_to_nx(g_data)
        uuids = _nx_node_uuids(g_nx) if _CIC_WORKER_CTX["compute_cic"] else None
        g_dgl = dgl.from_networkx(g_nx, node_attrs=['type'], edge_attrs=['type'])
        del g_nx

        g_dgl = transform_graph_with_cic(
            g_dgl,
            _CIC_WORKER_CTX["node_feature_dim"],
            _CIC_WORKER_CTX["edge_feature_dim"],
            _CIC_WORKER_CTX["cic_scores"],
            node_uuids=uuids,
        )

        with open(out_path, 'wb') as f:
            pkl.dump(g_dgl, f)

        del g_dgl
        del uuids
        return idx, True, None
    except Exception as e:
        return idx, False, str(e)


class StreamspotDataset(dgl.data.DGLDataset):
    def process(self):
        pass

    def __init__(self, name):
        super(StreamspotDataset, self).__init__(name=name)
        if name == 'streamspot':
            path = './data/streamspot'
            num_graphs = 600
            self.graphs = []
            self.labels = []
            print('Loading {} dataset...'.format(name))
            for i in tqdm(range(num_graphs)):
                idx = i
                with open('{}/{}.json'.format(path, str(idx + 1)), 'r', encoding='utf-8') as f:
                    g_json = json.load(f)
                g = dgl.from_networkx(
                    nx.node_link_graph(g_json),
                    node_attrs=['type'],
                    edge_attrs=['type']
                )
                self.graphs.append(g)
                if 300 <= idx <= 399:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


class WgetDataset(dgl.data.DGLDataset):
    def process(self):
        pass

    def __init__(self, name):
        super(WgetDataset, self).__init__(name=name)
        if name == 'wget':
            path = './data/wget/final'
            num_graphs = 150
            self.graphs = []
            self.labels = []
            print('Loading {} dataset...'.format(name))
            for i in tqdm(range(num_graphs)):
                idx = i
                with open('{}/{}.json'.format(path, str(idx)), 'r', encoding='utf-8') as f:
                    g_json = json.load(f)
                g = dgl.from_networkx(
                    nx.node_link_graph(g_json),
                    node_attrs=['type'],
                    edge_attrs=['type']
                )
                self.graphs.append(g)
                if 0 <= idx <= 24:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


def load_rawdata(name):
    if name == 'streamspot':
        path = './data/streamspot'
        if os.path.exists(path + '/graphs.pkl'):
            print('Loading processed {} dataset...'.format(name))
            with open(path + '/graphs.pkl', 'rb') as f:
                raw_data = pkl.load(f)
        else:
            raw_data = StreamspotDataset(name)
            with open(path + '/graphs.pkl', 'wb') as f:
                pkl.dump(raw_data, f)
    elif name == 'wget':
        path = './data/wget'
        if os.path.exists(path + '/graphs.pkl'):
            print('Loading processed {} dataset...'.format(name))
            with open(path + '/graphs.pkl', 'rb') as f:
                raw_data = pkl.load(f)
        else:
            raw_data = WgetDataset(name)
            with open(path + '/graphs.pkl', 'wb') as f:
                pkl.dump(raw_data, f)
    else:
        raise NotImplementedError
    return raw_data


def load_batch_level_dataset(dataset_name):
    dataset = load_rawdata(dataset_name)
    node_feature_dim = 0
    edge_feature_dim = 0
    for g, _ in dataset:
        node_feature_dim = max(node_feature_dim, g.ndata["type"].max().item())
        edge_feature_dim = max(edge_feature_dim, g.edata["type"].max().item())
    node_feature_dim += 1
    edge_feature_dim += 1
    full_dataset = [i for i in range(len(dataset))]
    train_dataset = [i for i in range(len(dataset)) if dataset[i][1] == 0]
    print('[n_graph, n_node_feat, n_edge_feat]: [{}, {}, {}]'.format(len(dataset), node_feature_dim, edge_feature_dim))

    return {'dataset': dataset,
            'train_index': train_dataset,
            'full_index': full_dataset,
            'n_feat': node_feature_dim,
            'e_feat': edge_feature_dim}


def transform_graph(g, node_feature_dim, edge_feature_dim):
    new_g = g.clone()
    new_g.ndata["attr"] = F.one_hot(g.ndata["type"].view(-1), num_classes=node_feature_dim).float()
    new_g.edata["attr"] = F.one_hot(g.edata["type"].view(-1), num_classes=edge_feature_dim).float()
    return new_g


def preload_entity_level_dataset(path):
    data_path = resolve_data_dir(path)
    if os.path.exists(os.path.join(data_path, 'metadata.json')):
        pass
    else:
        print('transforming')
        with open(os.path.join(data_path, 'train.pkl'), 'rb') as f:
            train_data = pkl.load(f)
        train_gs = [
            dgl.from_networkx(nx.node_link_graph(g), node_attrs=['type'], edge_attrs=['type'])
            for g in train_data
        ]
        print('transforming')
        with open(os.path.join(data_path, 'test.pkl'), 'rb') as f:
            test_data = pkl.load(f)
        test_gs = [
            dgl.from_networkx(nx.node_link_graph(g), node_attrs=['type'], edge_attrs=['type'])
            for g in test_data
        ]
        malicious_path = os.path.join(data_path, 'malicious.pkl')
        if os.path.exists(malicious_path):
            with open(malicious_path, 'rb') as f:
                malicious = pkl.load(f)
        else:
            malicious = None

        node_feature_dim = 0
        for g in train_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
        for g in test_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
        node_feature_dim += 1
        edge_feature_dim = 0
        for g in train_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)
        for g in test_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)
        edge_feature_dim += 1
        result_test_gs = []
        for g in test_gs:
            g = transform_graph(g, node_feature_dim, edge_feature_dim)
            result_test_gs.append(g)
        result_train_gs = []
        for g in train_gs:
            g = transform_graph(g, node_feature_dim, edge_feature_dim)
            result_train_gs.append(g)
        metadata = {
            'node_feature_dim': node_feature_dim,
            'edge_feature_dim': edge_feature_dim,
            'malicious': malicious,
            'n_train': len(result_train_gs),
            'n_test': len(result_test_gs)
        }
        with open(os.path.join(data_path, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        for i, g in enumerate(result_train_gs):
            with open(os.path.join(data_path, 'train{}.pkl'.format(i)), 'wb') as f:
                pkl.dump(g, f)
        for i, g in enumerate(result_test_gs):
            with open(os.path.join(data_path, 'test{}.pkl'.format(i)), 'wb') as f:
                pkl.dump(g, f)


def load_metadata(path):
    preload_entity_level_dataset(path)
    data_path = resolve_data_dir(path)
    with open(os.path.join(data_path, 'metadata.json'), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def load_entity_level_dataset(path, t, n):
    preload_entity_level_dataset(path)
    data_path = resolve_data_dir(path)
    with open(os.path.join(data_path, '{}{}.pkl'.format(t, n)), 'rb') as f:
        data = pkl.load(f)
    return data


# ============================================================================
# CIC 不变量集成 (Phase 1)
# ============================================================================

def load_cic_metadata(path: str) -> dict:
    """
    加载CIC相关的元数据文件
    
    Args:
        path: 数据集路径 (如 'theia', 'trace', 'cadets')
        
    Returns:
        包含entities, invariant_tracking, names, types的字典
    """
    data_path = resolve_data_dir(path)
    result = {}
    
    # 加载entities.pkl
    entities_path = os.path.join(data_path, 'entities.pkl')
    if os.path.exists(entities_path):
        with open(entities_path, 'rb') as f:
            result['entities'] = pkl.load(f)
        print(f'[CIC] 已加载 entities.pkl')
    else:
        result['entities'] = None
        print(f'[CIC] 警告: 未找到 entities.pkl')
    
    # 加载invariant_tracking.pkl
    invariant_path = os.path.join(data_path, 'invariant_tracking.pkl')
    if os.path.exists(invariant_path):
        with open(invariant_path, 'rb') as f:
            result['invariant_tracking'] = pkl.load(f)
        print(f'[CIC] 已加载 invariant_tracking.pkl')
    else:
        result['invariant_tracking'] = None
        print(f'[CIC] 警告: 未找到 invariant_tracking.pkl')
    
    # 加载names.pkl
    names_path = os.path.join(data_path, 'names.pkl')
    if os.path.exists(names_path):
        with open(names_path, 'rb') as f:
            result['names'] = pkl.load(f)
        print(f'[CIC] 已加载 names.pkl')
    else:
        result['names'] = None
    
    # 加载types.pkl
    types_path = os.path.join(data_path, 'types.pkl')
    if os.path.exists(types_path):
        with open(types_path, 'rb') as f:
            result['types'] = pkl.load(f)
        print(f'[CIC] 已加载 types.pkl')
    else:
        result['types'] = None
    
    return result


def load_cic_invariant_scores(path: str) -> dict:
    """
    计算并加载CIC不变量分数
    
    Args:
        path: 数据集路径
        
    Returns:
        uuid -> InvariantScores 的字典
    """
    try:
        from utils.cic_invariants import compute_and_save_cic_scores

        data_path = resolve_data_dir(path)
        all_scores = compute_and_save_cic_scores(data_path)

        print(f"[CIC] Loaded/computed CIC scores for {len(all_scores)} entities")
        return all_scores
    
    except FileNotFoundError as e:
        print(f'[CIC] 警告: {e}')
        return {}
    except ImportError as e:
        print(f'[CIC] 警告: 无法导入cic_invariants模块: {e}')
        return {}
    except Exception as e:
        print(f'[CIC] 警告: 计算CIC分数失败: {e}')
        return {}


def transform_graph_with_cic(g, node_feature_dim, edge_feature_dim, 
                              cic_scores=None, node_uuids=None, uuid_to_idx=None):
    """
    转换图并添加CIC不变量分数作为节点特征
    
    Args:
        g: DGL图
        node_feature_dim: 节点类型数量
        edge_feature_dim: 边类型数量
        cic_scores: uuid -> InvariantScores 的字典 (可选)
        uuid_to_idx: uuid -> 节点索引 的字典 (可选)
        
    Returns:
        添加了attr和cic_scores特征的DGL图
    """
    import torch
    
    new_g = g.clone()
    
    # 基础特征：节点/边类型的one-hot编码
    new_g.ndata["attr"] = F.one_hot(g.ndata["type"].view(-1), num_classes=node_feature_dim).float()
    new_g.edata["attr"] = F.one_hot(g.edata["type"].view(-1), num_classes=edge_feature_dim).float()
    
    # 添加CIC不变量分数特征
    # 注意：DGL 的 ndata 不适合直接存字符串 uuid；因此优先使用 node_uuids（与节点顺序对齐）。
    if cic_scores is not None:
        n_nodes = new_g.num_nodes()
        cic_features = torch.zeros(n_nodes, 4, dtype=torch.float32)  # 4个不变量

        if node_uuids is not None:
            for idx, uuid in enumerate(node_uuids):
                scores = cic_scores.get(uuid)
                if scores is None:
                    continue
                cic_features[idx] = torch.from_numpy(scores.to_vector()).to(torch.float32)
            new_g.ndata["cic_scores"] = cic_features
        elif uuid_to_idx is not None:
            # 兼容旧调用方式（不推荐：uuid_to_idx 需是“当前图内”的映射）
            for uuid, scores in cic_scores.items():
                idx = uuid_to_idx.get(uuid)
                if idx is None or idx >= n_nodes:
                    continue
                cic_features[idx] = torch.from_numpy(scores.to_vector()).to(torch.float32)
            new_g.ndata["cic_scores"] = cic_features
    
    return new_g


def _iter_graphs_from_pkl(pkl_path: str):
    """
    生成器：从 pkl 文件逐个 yield 图数据，减少峰值内存。
    返回 (index, graph_data) 元组。
    """
    with open(pkl_path, 'rb') as f:
        data_list = pkl.load(f)
    
    # 逐个 yield 并释放引用
    for i in range(len(data_list)):
        g_data = data_list[i]
        data_list[i] = None  # 释放引用
        yield i, g_data
        del g_data
        
        # 每处理 100 个图触发一次 gc
        if i > 0 and i % 100 == 0:
            gc.collect()
    
    del data_list
    gc.collect()


def preload_entity_level_dataset_with_cic(path: str, compute_cic: bool = True, workers: int = 1):
    """
    Preload entity-level dataset and optionally attach CIC scores.
    内存优化版：使用生成器逐个处理图，避免峰值内存过高。
    """
    data_path = resolve_data_dir(path)
    metadata_path = os.path.join(data_path, 'metadata_cic.json')

    if os.path.exists(metadata_path):
        print('[CIC] metadata_cic.json already exists; skip preprocessing')
        return

    print(f'[CIC] 开始预处理数据集: {data_path}')

    # Load raw graphs (node_link_data list)
    train_pkl = os.path.join(data_path, 'train.pkl')
    test_pkl = os.path.join(data_path, 'test.pkl')
    malicious_pkl = os.path.join(data_path, 'malicious.pkl')

    if not os.path.exists(train_pkl) or not os.path.exists(test_pkl):
        print(f'[CIC] Missing {train_pkl} or {test_pkl}; run trace_parser.py first')
        return

    node_feature_dim, edge_feature_dim, malicious = _resolve_feature_dims_from_metadata(data_path)
    if node_feature_dim is None or edge_feature_dim is None:
        node_dim, edge_dim = _resolve_feature_dims_from_type_mappings(data_path)
        if node_feature_dim is None:
            node_feature_dim = node_dim
        if edge_feature_dim is None:
            edge_feature_dim = edge_dim

    if malicious is None:
        if os.path.exists(malicious_pkl):
            with open(malicious_pkl, 'rb') as f:
                malicious = pkl.load(f)
        else:
            malicious = None

    dims_ready = node_feature_dim is not None and edge_feature_dim is not None
    if not dims_ready:
        print('[CIC] 计算特征维度...（第一遍扫描 train.pkl）')
        node_feature_dim = node_feature_dim or 0
        edge_feature_dim = edge_feature_dim or 0
        
        # 使用生成器遍历，减少峰值内存
        for _, g_data in _iter_graphs_from_pkl(train_pkl):
            for n in g_data.get("nodes", []):
                if "type" in n:
                    node_feature_dim = max(node_feature_dim, int(n["type"]))
            for e in g_data.get("links", []):
                if "type" in e:
                    edge_feature_dim = max(edge_feature_dim, int(e["type"]))
        
        gc.collect()
        
        print('[CIC] 计算特征维度...（第一遍扫描 test.pkl）')
        for _, g_data in _iter_graphs_from_pkl(test_pkl):
            for n in g_data.get("nodes", []):
                if "type" in n:
                    node_feature_dim = max(node_feature_dim, int(n["type"]))
            for e in g_data.get("links", []):
                if "type" in e:
                    edge_feature_dim = max(edge_feature_dim, int(e["type"]))
        
        gc.collect()
        node_feature_dim += 1
        edge_feature_dim += 1
    
    print(f'[CIC] 特征维度: node={node_feature_dim}, edge={edge_feature_dim}')

    # Load CIC scores
    cic_scores = {}
    if compute_cic:
        print('[CIC] 加载/计算 CIC 分数...')
        cic_scores = load_cic_invariant_scores(path)
        gc.collect()

    # Transform and persist per-graph to limit peak memory
    def process_split_streaming(split_name, pkl_path):
        """流式处理：逐个图读取、转换、保存，避免一次性加载全部图"""
        n_split = 0
        n_skipped = 0

        if workers <= 1:
            for i, g_data in tqdm(_iter_graphs_from_pkl(pkl_path), desc=f'处理 {split_name} 图'):
                # 检查是否已处理过
                out_path = os.path.join(data_path, f'{split_name}_cic{i}.pkl')
                if os.path.exists(out_path):
                    n_split += 1
                    n_skipped += 1
                    continue
                
                try:
                    g_nx = _node_link_to_nx(g_data)
                    uuids = _nx_node_uuids(g_nx) if compute_cic else None
                    g_dgl = dgl.from_networkx(g_nx, node_attrs=['type'], edge_attrs=['type'])

                    # 立即释放 networkx 图
                    del g_nx

                    g_dgl = transform_graph_with_cic(
                        g_dgl, node_feature_dim, edge_feature_dim, cic_scores, node_uuids=uuids
                    )

                    with open(out_path, 'wb') as f:
                        pkl.dump(g_dgl, f)

                    # 立即释放 DGL 图
                    del g_dgl
                    del uuids

                    n_split += 1

                except Exception as e:
                    print(f'[CIC] 警告: 处理 {split_name} 图 {i} 时出错: {e}')
                    continue

            if n_skipped > 0:
                print(f'[CIC] 跳过 {n_skipped} 个已处理的 {split_name} 图')
            gc.collect()
            return n_split

        ctx = _get_mp_context()
        if ctx.get_start_method() != "fork":
            print("[CIC] 当前平台不支持 fork，多进程会复制大内存，改为单进程执行")
            for i, g_data in tqdm(_iter_graphs_from_pkl(pkl_path), desc=f'处理 {split_name} 图'):
                # 检查是否已处理过
                out_path = os.path.join(data_path, f'{split_name}_cic{i}.pkl')
                if os.path.exists(out_path):
                    n_split += 1
                    n_skipped += 1
                    continue
                
                try:
                    g_nx = _node_link_to_nx(g_data)
                    uuids = _nx_node_uuids(g_nx) if compute_cic else None
                    g_dgl = dgl.from_networkx(g_nx, node_attrs=['type'], edge_attrs=['type'])

                    del g_nx

                    g_dgl = transform_graph_with_cic(
                        g_dgl, node_feature_dim, edge_feature_dim, cic_scores, node_uuids=uuids
                    )

                    with open(out_path, 'wb') as f:
                        pkl.dump(g_dgl, f)

                    del g_dgl
                    del uuids

                    n_split += 1
                except Exception as e:
                    print(f'[CIC] 警告: 处理 {split_name} 图 {i} 时出错: {e}')
                    continue

            if n_skipped > 0:
                print(f'[CIC] 跳过 {n_skipped} 个已处理的 {split_name} 图')
            gc.collect()
            return n_split

        global _CIC_WORKER_CTX
        _CIC_WORKER_CTX = {
            "node_feature_dim": node_feature_dim,
            "edge_feature_dim": edge_feature_dim,
            "cic_scores": cic_scores,
            "compute_cic": compute_cic,
        }

        tasks = ((i, g_data, split_name, data_path) for i, g_data in _iter_graphs_from_pkl(pkl_path))

        with ctx.Pool(processes=workers, initializer=_init_cic_worker) as pool:
            for idx, ok, err in tqdm(pool.imap_unordered(_process_cic_graph, tasks),
                                     desc=f'处理 {split_name} 图'):
                if ok:
                    n_split += 1
                else:
                    print(f'[CIC] 警告: 处理 {split_name} 图 {idx} 时出错: {err}')

        gc.collect()
        return n_split

    print('[CIC] 处理训练集...')
    n_train = process_split_streaming('train', train_pkl)
    gc.collect()

    print('[CIC] 处理测试集...')
    n_test = process_split_streaming('test', test_pkl)
    gc.collect()

    has_cic_scores = compute_cic and len(cic_scores) > 0

    # 释放 CIC 分数（已不再需要）
    del cic_scores
    gc.collect()

    # Save metadata
    metadata = {
        'node_feature_dim': node_feature_dim,
        'edge_feature_dim': edge_feature_dim,
        'cic_feature_dim': 4,
        'malicious': malicious,
        'n_train': n_train,
        'n_test': n_test,
        'has_cic_scores': has_cic_scores,
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f)

    print(f'[CIC] 预处理完成: {n_train} 训练图, {n_test} 测试图')


def load_metadata_with_cic(path: str) -> dict:
    """加载带有CIC信息的metadata"""
    data_path = resolve_data_dir(path)
    
    # 优先加载CIC增强的metadata
    cic_metadata_path = data_path + '/metadata_cic.json'
    if os.path.exists(cic_metadata_path):
        with open(cic_metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 回退到标准metadata
    return load_metadata(path)


def load_entity_level_dataset_with_cic(path: str, split: str, n: int):
    """
    加载带有CIC分数的实体级数据集
    
    Args:
        path: 数据集路径
        split: 'train' 或 'test'
        n: 图索引
        
    Returns:
        DGL图
    """
    data_path = resolve_data_dir(path)
    
    # 优先加载CIC增强的图
    cic_path = data_path + '/{}_cic{}.pkl'.format(split, n)
    if os.path.exists(cic_path):
        with open(cic_path, 'rb') as f:
            return pkl.load(f)
    
    # 回退到标准图
    return load_entity_level_dataset(path, split, n)
