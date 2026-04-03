"""
Minimal Explanation Subgraph (MES) Builder

实现研究路线中的4阶段算法：
1. 初始化: 选取Top-K高分元素R并将触发不变量的证据模板并入候选，形成S₀
2. 贪心扩张: 选择"越扩展越值"的边，收益 = 增量奖赏/增量成本
3. 证据完备扩张: 添加让异常因子成立的最小证据集
4. 后剪枝: 从叶子节点删除不影响连通性和证据完备性的元素

公式:
- 节点优先级: p_x = α·CIC(x) + (1-α)·anom(x) ∈ [0,1]
- 增量奖赏: ΔP(Y|S) = Σp_x (x∈Y∪endpoints(Y)) - Σp_x (x∈S∩(Y∪endpoints(Y)))
- 增量成本: ΔC(Y|S) = λ_E·Σc_e + λ_V·|{v∈Y ∧ v∉V_S}|
- 收益: gain = ΔP / ΔC
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set
from collections import deque
import numpy as np
import torch
import networkx as nx
import dgl

from model.explanation import ExplanationNode, ExplanationEdge, ExplanationSubgraph


class MinimalExplanationBuilder:
    """
    最小解释子图构建器
    
    实现研究路线中的4阶段算法：
    1. 初始化: Top-K种子节点
    2. 贪心扩张: 收益/成本最大化
    3. 证据完备扩张: 添加不变量证据
    4. 后剪枝: 删除冗余叶节点
    """
    
    NODE_TYPE_NAMES = {0: 'subject', 1: 'file', 2: 'netflow', 3: 'memory', 4: 'principal'}
    
    def __init__(self,
                 graph: dgl.DGLGraph,
                 cic_scores: torch.Tensor,
                 anomaly_scores: torch.Tensor,
                 alpha: float = 0.5,
                 lambda_e: float = 1.0,
                 lambda_v: float = 1.0,
                 edge_cost: float = 1.0,
                 cic_weights: Optional[List[float]] = None,
                 anomaly_threshold: float = 0.5,
                 verbose: bool = False,
                 names_map: Optional[Dict[int, str]] = None,
                 types_map: Optional[Dict[int, int]] = None):
        """
        Args:
            graph: DGL图
            cic_scores: CIC分数 [n_nodes, 4] 或 [n_nodes]
            anomaly_scores: 异常分数 [n_nodes]
            alpha: CIC与异常分数的融合权重 (研究路线公式中的α)
            lambda_e: 边成本权重 (研究路线公式中的λ_E)
            lambda_v: 新节点成本权重 (研究路线公式中的λ_V)
            edge_cost: 每条边的基础成本 c_e
            names_map: 节点名称映射
            types_map: 节点类型映射
        """
        # Explanation is CPU- and python-loop heavy; keep graph on CPU.
        self.graph = graph.to("cpu") if getattr(graph, "device", None) is not None and str(graph.device) != "cpu" else graph
        self.n_nodes = int(self.graph.num_nodes())
        self.n_edges = int(self.graph.num_edges())
        
        # 计算节点优先级 p_x = α·CIC(x) + (1-α)·anom(x)
        if cic_scores.dim() == 2:
            # 4维CIC分数，使用风险放大公式计算总分
            if cic_weights is None:
                w = [0.25, 0.25, 0.25, 0.25]
            else:
                if len(cic_weights) != 4:
                    raise ValueError(f"cic_weights must have 4 elements, got {len(cic_weights)}")
                w = [float(x) for x in cic_weights]
            weights = torch.tensor(w, device=cic_scores.device, dtype=torch.float32)
            cic_total = 1.0 - (1.0 - weights.unsqueeze(0) * cic_scores.clamp(0, 1)).prod(dim=-1)
        else:
            cic_total = cic_scores.clamp(0, 1)
        
        cic_total = torch.nan_to_num(cic_total, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        anomaly_scores = torch.nan_to_num(anomaly_scores, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        
        # p_x = α·CIC(x) + (1-α)·anom(x)
        self.node_priority = (alpha * cic_total + (1 - alpha) * anomaly_scores).detach().cpu().numpy().astype(np.float32)
        self.cic_scores = cic_scores.detach().cpu().numpy().astype(np.float32) if cic_scores.dim() == 2 else None
        self.cic_total = cic_total.detach().cpu().numpy().astype(np.float32)
        self.anomaly_scores_np = anomaly_scores.detach().cpu().numpy().astype(np.float32)
        
        self.alpha = alpha
        self.lambda_e = lambda_e
        self.lambda_v = lambda_v
        self.edge_cost = edge_cost
        self.anomaly_threshold = float(anomaly_threshold)
        self.verbose = bool(verbose)
        
        self.names_map = names_map or {}
        self.types_map = types_map or {}

        # Cache endpoints only for edges we touch (avoid storing full adjacency for large graphs).
        self._edge_endpoints: Dict[int, Tuple[int, int]] = {}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _iter_out_edges(self, node: int) -> List[Tuple[int, int, int]]:
        """Return (src, dst, eid) for out edges of `node`."""
        u, v, eids = self.graph.out_edges(int(node), form="all")
        if eids.numel() == 0:
            return []
        src = int(node)
        vs = v.tolist()
        es = eids.tolist()
        return [(src, int(dst), int(eid)) for dst, eid in zip(vs, es)]

    def _iter_in_edges(self, node: int) -> List[Tuple[int, int, int]]:
        """Return (src, dst, eid) for in edges of `node`."""
        u, v, eids = self.graph.in_edges(int(node), form="all")
        if eids.numel() == 0:
            return []
        dst = int(node)
        us = u.tolist()
        es = eids.tolist()
        return [(int(src), dst, int(eid)) for src, eid in zip(us, es)]

    def _edge_uv(self, eid: int) -> Tuple[int, int]:
        eid = int(eid)
        if eid in self._edge_endpoints:
            return self._edge_endpoints[eid]
        u, v = self.graph.find_edges(torch.tensor([eid], dtype=torch.int64))
        src = int(u[0].item())
        dst = int(v[0].item())
        self._edge_endpoints[eid] = (src, dst)
        return src, dst
    
    def _get_node_priority(self, node_id: int) -> float:
        """获取节点优先级 p_x"""
        if 0 <= node_id < len(self.node_priority):
            return float(self.node_priority[node_id])
        return 0.0
    
    def _compute_marginal_gain(self,
                               edge_triplet: Tuple[int, int, int],
                               current_nodes: Set[int]) -> float:
        """
        单步增量收益：gain = ΔP / ΔC

        这里按“每次只增加一条边”的贪心策略计算。
        """
        s, d, _ = edge_triplet
        new_nodes = {int(s), int(d)}
        truly_new = new_nodes - current_nodes
        reward = sum(self._get_node_priority(n) for n in truly_new)
        cost = self.lambda_e * float(self.edge_cost) + self.lambda_v * float(len(truly_new)) + 1e-8
        return float(reward) / float(cost)
    
    def _get_candidate_edges(self, 
                               current_nodes: Set[int],
                               current_edges: Set[int]) -> List[Tuple[int, int, int]]:
        """获取可扩展的候选边（frontier边），返回 (src, dst, eid) 列表。"""
        seen: Set[int] = set()
        candidates: List[Tuple[int, int, int]] = []
        for node in current_nodes:
            for s, d, eid in self._iter_out_edges(node):
                if eid in current_edges or eid in seen:
                    continue
                seen.add(eid)
                self._edge_endpoints[eid] = (int(s), int(d))
                candidates.append((int(s), int(d), int(eid)))
            for s, d, eid in self._iter_in_edges(node):
                if eid in current_edges or eid in seen:
                    continue
                seen.add(eid)
                self._edge_endpoints[eid] = (int(s), int(d))
                candidates.append((int(s), int(d), int(eid)))
        return candidates
    
    def _is_connected(self, nodes: Set[int], edges: Set[int]) -> bool:
        """检查子图是否连通（无向意义上）"""
        if len(nodes) <= 1:
            return True
        
        adj: Dict[int, Set[int]] = {n: set() for n in nodes}
        for eidx in edges:
            s, d = self._edge_uv(int(eidx))
            if s in nodes and d in nodes:
                adj[s].add(d)
                adj[d].add(s)
        
        start = next(iter(nodes))
        visited = {start}
        queue: deque[int] = deque([start])
        while queue:
            curr = queue.popleft()
            for neighbor in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(nodes)

    def _component_map(self, nodes: Set[int], edges: Set[int]) -> Dict[int, int]:
        """返回 node -> component_id（无向意义上的连通分量）。"""
        if not nodes:
            return {}
        adj: Dict[int, Set[int]] = {n: set() for n in nodes}
        for eidx in edges:
            s, d = self._edge_uv(int(eidx))
            if s in nodes and d in nodes:
                adj[s].add(d)
                adj[d].add(s)
        comp: Dict[int, int] = {}
        cid = 0
        for n in nodes:
            if n in comp:
                continue
            cid += 1
            q: deque[int] = deque([n])
            comp[n] = cid
            while q:
                cur = q.popleft()
                for nb in adj.get(cur, []):
                    if nb not in comp:
                        comp[nb] = cid
                        q.append(nb)
        return comp
    
    def _get_invariant_evidence_edges(self, node: int) -> List[Tuple[int, int, int]]:
        """
        获取节点不变量成立所需的证据边
        
        对于高CIC分数的节点，需要保留其证据：
        - I_reach: 需要看到之前的访问边（入边）
        - I_creator: 需要看到创建者边（入边）
        - I_timing: 需要看到时间相关的边
        - I_alias: 需要看到别名关系边
        """
        evidence_edges: List[Tuple[int, int, int]] = []

        # 入边作为主要证据（I_reach, I_creator 的证据通常来自入边）
        for s, d, eid in self._iter_in_edges(node):
            if self._get_node_priority(int(s)) > 0.2:
                evidence_edges.append((int(s), int(d), int(eid)))

        # subject 类型节点的出边也可能是证据
        node_type = self.types_map.get(int(node), -1)
        if node_type == 0:  # subject
            for s, d, eid in self._iter_out_edges(node):
                if self._get_node_priority(int(d)) > 0.3:
                    evidence_edges.append((int(s), int(d), int(eid)))

        # Deduplicate by eid
        out: List[Tuple[int, int, int]] = []
        seen: Set[int] = set()
        for s, d, eid in evidence_edges:
            if eid in seen:
                continue
            seen.add(eid)
            self._edge_endpoints[eid] = (int(s), int(d))
            out.append((int(s), int(d), int(eid)))
        return out
    
    def build_minimal_subgraph(self,
                                 seed_nodes: List[int],
                                 max_nodes: int = 50,
                                 max_edges: int = 100,
                                 min_gain_threshold: float = 0.01,
                                 ensure_connected: bool = True) -> ExplanationSubgraph:
        """
        构建最小解释子图 (Minimal Explanation Subgraph)
        
        Args:
            seed_nodes: 种子节点列表（Top-K异常节点）- 研究路线中的R集合
            max_nodes: 最大节点数
            max_edges: 最大边数
            min_gain_threshold: 最小收益阈值（当gain < threshold时停止扩张）
            ensure_connected: 是否确保子图连通
            
        Returns:
            ExplanationSubgraph对象 - 研究路线中的S*
        """
        # ========== 阶段1: 初始化 S₀ ==========
        # S₀ = 种子节点集合，此时覆盖了主要异常因子，但可能不连通
        current_nodes: Set[int] = set(seed_nodes)
        current_edges: Set[int] = set()

        # 添加种子节点之间的边（只扫描种子邻域，避免 O(E)）
        seed_set = set(int(n) for n in seed_nodes)
        for n in list(seed_set):
            for s, d, eid in self._iter_out_edges(n):
                if int(d) in seed_set:
                    current_edges.add(int(eid))
                    self._edge_endpoints[int(eid)] = (int(s), int(d))
            for s, d, eid in self._iter_in_edges(n):
                if int(s) in seed_set:
                    current_edges.add(int(eid))
                    self._edge_endpoints[int(eid)] = (int(s), int(d))

        self._log(f"[MES] 阶段1完成: {len(current_nodes)} 节点, {len(current_edges)} 边")
        
        # ========== 阶段2: 贪心扩张 ==========
        # 每次选择收益最高的边：gain = ΔP / ΔC
        iteration = 0
        max_iterations = max_nodes * 2
        
        while len(current_nodes) < max_nodes and len(current_edges) < max_edges and iteration < max_iterations:
            iteration += 1
            
            candidates = self._get_candidate_edges(current_nodes, current_edges)
            if not candidates:
                break
            
            best_edge: Optional[Tuple[int, int, int]] = None
            best_gain = -float('inf')

            for triplet in candidates:
                gain = self._compute_marginal_gain(triplet, current_nodes)
                if gain > best_gain:
                    best_gain, best_edge = gain, triplet
            
            if best_gain < min_gain_threshold:
                break
            
            if best_edge is not None:
                s, d, eid = best_edge
                current_edges.add(int(eid))
                self._edge_endpoints[int(eid)] = (int(s), int(d))
                current_nodes.add(int(s))
                current_nodes.add(int(d))

        # 若要求连通但仍不连通，则在预算内继续扩张直到连通（忽略收益阈值）
        if ensure_connected and not self._is_connected(current_nodes, current_edges):
            safety_iters = max_nodes * 4
            while (not self._is_connected(current_nodes, current_edges)) and len(current_nodes) < max_nodes and len(current_edges) < max_edges and safety_iters > 0:
                safety_iters -= 1
                candidates = self._get_candidate_edges(current_nodes, current_edges)
                if not candidates:
                    break
                comp = self._component_map(current_nodes, current_edges)
                connecting = [
                    t for t in candidates
                    if (int(t[0]) in comp and int(t[1]) in comp and comp[int(t[0])] != comp[int(t[1])])
                ]
                pool = connecting if connecting else candidates
                best_edge = max(pool, key=lambda t: self._compute_marginal_gain(t, current_nodes))
                s, d, eid = best_edge
                current_edges.add(int(eid))
                self._edge_endpoints[int(eid)] = (int(s), int(d))
                current_nodes.add(int(s))
                current_nodes.add(int(d))

        self._log(f"[MES] 阶段2完成: {len(current_nodes)} 节点, {len(current_edges)} 边")
        
        # ========== 阶段3: 证据完备扩张 ==========
        # 对于每个高CIC分数节点，添加其证据边
        evidence_added = 0
        evidence_nodes: Set[int] = set()
        for node in list(current_nodes):
            if node < len(self.cic_total) and float(self.cic_total[node]) > 0.5:
                for s, d, eid in self._get_invariant_evidence_edges(int(node)):
                    if int(eid) not in current_edges and len(current_edges) < max_edges:
                        current_edges.add(int(eid))
                        self._edge_endpoints[int(eid)] = (int(s), int(d))
                        current_nodes.add(int(s))
                        current_nodes.add(int(d))
                        evidence_nodes.add(int(s))
                        evidence_nodes.add(int(d))
                        evidence_added += 1

        self._log(f"[MES] 阶段3完成: 添加了 {evidence_added} 条证据边")
        
        # ========== 阶段4: 后剪枝 ==========
        # 从叶子节点开始删除不影响连通性和证据完备性的低优先级节点
        seed_set = set(int(n) for n in seed_nodes)
        pruned_count = 0
        
        if ensure_connected:
            changed = True
            while changed and len(current_nodes) > len(seed_nodes):
                changed = False
                
                # 计算度数
                degree = {n: 0 for n in current_nodes}
                for eidx in current_edges:
                    s, d = self._edge_uv(int(eidx))
                    if s in degree:
                        degree[s] += 1
                    if d in degree:
                        degree[d] += 1
                
                # 按优先级升序排列叶节点
                leaf_nodes = sorted(
                    [n for n, deg in degree.items() if deg <= 1 and n not in seed_set and n not in evidence_nodes],
                    key=lambda n: self._get_node_priority(n)
                )
                
                for leaf in leaf_nodes:
                    # 额外保护：不要删除与“高CIC节点”直接相邻的节点（较保守）。
                    high_cic_neighbor = False
                    for s, d, _ in self._iter_out_edges(int(leaf)):
                        if int(d) in current_nodes and int(d) < len(self.cic_total) and float(self.cic_total[int(d)]) > 0.7:
                            high_cic_neighbor = True
                            break
                    if not high_cic_neighbor:
                        for s, d, _ in self._iter_in_edges(int(leaf)):
                            if int(s) in current_nodes and int(s) < len(self.cic_total) and float(self.cic_total[int(s)]) > 0.7:
                                high_cic_neighbor = True
                                break
                    if high_cic_neighbor:
                        continue
                    
                    # 检查删除后是否仍连通
                    test_nodes = current_nodes - {leaf}
                    test_edges: Set[int] = set()
                    for e in current_edges:
                        s, d = self._edge_uv(int(e))
                        if int(s) == int(leaf) or int(d) == int(leaf):
                            continue
                        test_edges.add(int(e))
                    
                    if len(test_nodes) >= len(seed_nodes) and self._is_connected(test_nodes, test_edges):
                        current_nodes, current_edges = test_nodes, test_edges
                        pruned_count += 1
                        changed = True
                        break
        
        self._log(f"[MES] 阶段4完成: 剪枝了 {pruned_count} 个节点")
        self._log(f"[MES] 最终结果: {len(current_nodes)} 节点, {len(current_edges)} 边")
        
        if ensure_connected and not self._is_connected(current_nodes, current_edges):
            self._log("[MES] 警告: 在预算约束下子图仍未完全连通（可增大 max_nodes/max_edges 或降低 min_gain_threshold）")
        return self._build_explanation_subgraph(current_nodes, current_edges, seed_nodes[0] if seed_nodes else 0)
    
    def _build_explanation_subgraph(self, 
                                      nodes: Set[int], 
                                      edges: Set[int], 
                                      center_node: int) -> ExplanationSubgraph:
        """构建ExplanationSubgraph对象"""
        exp_nodes = []
        for nid in sorted(nodes):
            nid = int(nid)
            display = self.names_map.get(nid, f"node_{nid}")
            cic_list = list(self.cic_scores[nid]) if self.cic_scores is not None and nid < len(self.cic_scores) else [0.0]*4
            
            exp_nodes.append(ExplanationNode(
                node_id=nid,
                uuid=display,
                node_type=self.NODE_TYPE_NAMES.get(self.types_map.get(nid, 0), 'unknown'),
                name=display,
                anomaly_score=float(self.anomaly_scores_np[nid]) if nid < len(self.anomaly_scores_np) else 0.0,
                cic_scores=cic_list,
                is_anomaly=self.anomaly_scores_np[nid] > float(self.anomaly_threshold) if nid < len(self.anomaly_scores_np) else False,
            ))
        
        exp_edges = []
        for eidx in sorted(edges):
            eidx = int(eidx)
            s, d = self._edge_uv(eidx)
            importance = (self._get_node_priority(s) + self._get_node_priority(d)) / 2.0
            
            exp_edges.append(ExplanationEdge(
                src_id=s,
                dst_id=d,
                importance=importance,
                is_attack_edge=(self.anomaly_scores_np[s] > float(self.anomaly_threshold) and self.anomaly_scores_np[d] > float(self.anomaly_threshold))
                              if s < len(self.anomaly_scores_np) and d < len(self.anomaly_scores_np) else False,
            ))
        
        total_score = sum(self._get_node_priority(n) for n in nodes) / max(len(nodes), 1)
        
        subgraph = ExplanationSubgraph(
            center_node=int(center_node),
            nodes=exp_nodes,
            edges=exp_edges,
            k_hop=-1,  # 不适用于MES
            total_anomaly_score=total_score,
        )
        
        self._find_attack_path(subgraph)
        
        return subgraph
    
    def _find_attack_path(self, subgraph: ExplanationSubgraph) -> None:
        """在子图中找攻击路径（最重要路径）"""
        if not subgraph.edges or len(subgraph.nodes) < 2:
            return
        
        G = nx.DiGraph()
        for edge in subgraph.edges:
            G.add_edge(edge.src_id, edge.dst_id, weight=1.0 - edge.importance + 0.01)
        
        anomaly_nodes = sorted(
            [(n.node_id, n.anomaly_score) for n in subgraph.nodes if n.is_anomaly],
            key=lambda x: x[1], reverse=True
        )
        
        if len(anomaly_nodes) >= 2:
            try:
                path = nx.shortest_path(G, source=anomaly_nodes[0][0], target=anomaly_nodes[-1][0], weight='weight')
                subgraph.attack_path = path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass


def build_minimal_explanation(graph: dgl.DGLGraph,
                              cic_scores: torch.Tensor,
                              anomaly_scores: torch.Tensor,
                              top_k: int = 5,
                              max_nodes: int = 50,
                              max_edges: int = 100,
                              alpha: float = 0.5,
                              lambda_e: float = 1.0,
                              lambda_v: float = 1.0,
                              seed_strategy: str = "priority",
                              cic_weights: Optional[List[float]] = None,
                              anomaly_threshold: float = 0.5,
                              verbose: bool = False,
                              **kwargs) -> ExplanationSubgraph:
    """
    便捷函数：构建最小解释子图
    
    Args:
        graph: DGL图
        cic_scores: CIC分数 [n_nodes, 4]
        anomaly_scores: 异常分数 [n_nodes]
        top_k: 种子节点数量
        max_nodes: 最大节点数
        max_edges: 最大边数
        alpha: CIC权重 (公式中的α)
        lambda_e: 边成本权重
        lambda_v: 节点成本权重
        seed_strategy: 种子选择策略：'priority'（默认，使用 p_x）或 'anomaly'（仅用 anomaly_scores）
        cic_weights: CIC 四维权重（默认 [0.25,0.25,0.25,0.25]）
        anomaly_threshold: 判定异常节点/攻击边的阈值（默认 0.5）
        verbose: 是否打印构建过程日志
        
    Returns:
        ExplanationSubgraph对象
    """
    builder = MinimalExplanationBuilder(
        graph=graph,
        cic_scores=cic_scores,
        anomaly_scores=anomaly_scores,
        alpha=alpha,
        lambda_e=lambda_e,
        lambda_v=lambda_v,
        cic_weights=cic_weights,
        anomaly_threshold=anomaly_threshold,
        verbose=verbose,
        **kwargs
    )
    
    # 选取Top-K种子节点作为 R 集合（研究路线更偏向使用融合优先级 p_x）
    if seed_strategy not in {"priority", "anomaly"}:
        raise ValueError(f"seed_strategy must be 'priority' or 'anomaly', got: {seed_strategy}")
    if seed_strategy == "priority":
        scores = torch.from_numpy(builder.node_priority)
    else:
        scores = anomaly_scores.detach().cpu().reshape(-1)
    k = min(int(top_k), int(scores.numel()))
    _, indices = torch.topk(scores, k)
    seed_nodes = [int(x) for x in indices.tolist()]
    
    return builder.build_minimal_subgraph(
        seed_nodes=seed_nodes,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )
