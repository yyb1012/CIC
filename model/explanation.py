"""
解释子图构建模块 (Explanation Subgraph Construction)

Phase 4 实现:
- 基于注意力权重和CIC分数构建k-hop解释子图
- 计算边重要性
- 生成攻击路径并可视化
- 社区发现辅助异常定位

与以下模块对接:
- utils/cic_invariants.py: InvariantScores
- model/fusion.py: AnomalyScorer融合分数
- model/gat.py: 注意力权重

参考: attack_investigation.py 的可视化效果
"""

import os
import pickle as pkl
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import math
import statistics

import torch
import torch.nn.functional as F
import networkx as nx
import dgl


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ExplanationNode:
    """解释子图中的节点信息"""
    node_id: int
    uuid: str = ""
    node_type: str = ""  # 'subject', 'file', 'netflow', 'memory'
    name: str = ""
    anomaly_score: float = 0.0
    cic_scores: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    is_anomaly: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationEdge:
    """解释子图中的边信息"""
    src_id: int
    dst_id: int
    edge_type: str = ""
    timestamp: int = 0
    importance: float = 0.0
    attention_weight: float = 0.0
    is_attack_edge: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationSubgraph:
    """完整的解释子图"""
    center_node: int
    nodes: List[ExplanationNode]
    edges: List[ExplanationEdge]
    k_hop: int = 2
    community_id: int = -1
    total_anomaly_score: float = 0.0
    attack_path: List[int] = field(default_factory=list)


# ============================================================================
# 解释子图构建器
# ============================================================================

class ExplanationSubgraphBuilder:
    """
    解释子图构建器
    
    基于CIC分数、注意力权重和重构误差构建可解释的异常子图。
    """
    
    # 节点类型映射 (与trace_parser.py保持一致)
    NODE_TYPE_NAMES = {
        0: 'subject',
        1: 'file', 
        2: 'netflow',
        3: 'memory',
        4: 'principal',
    }
    
    def __init__(self,
                 graph: dgl.DGLGraph,
                 node_embeddings: Optional[torch.Tensor] = None,
                 attention_weights: Optional[torch.Tensor] = None,
                 cic_scores: Optional[torch.Tensor] = None,
                 cic_weights: Optional[torch.Tensor] = None,
                 anomaly_scores: Optional[torch.Tensor] = None,
                 recon_errors: Optional[torch.Tensor] = None,
                 names_map: Optional[Dict[int, str]] = None,
                 types_map: Optional[Dict[int, int]] = None):
        """
        初始化解释子图构建器
        
        Args:
            graph: DGL图
            node_embeddings: 节点嵌入 [n_nodes, hidden_dim]
            attention_weights: 注意力权重 [n_edges, n_heads] 或 [n_edges]
            cic_scores: CIC分数 [n_nodes, 4]
            cic_weights: CIC融合权重（可选，4维；用于计算CIC总分时与fusion保持一致）
            anomaly_scores: 异常分数 [n_nodes]
            recon_errors: 重构误差 [n_nodes]
            names_map: 节点ID到名称的映射
            types_map: 节点ID到类型的映射
        """
        self.graph = graph
        self.node_embeddings = node_embeddings
        self.attention_weights = attention_weights
        self.cic_scores = cic_scores
        self.cic_weights = cic_weights
        self.anomaly_scores = (
            torch.nan_to_num(anomaly_scores, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            if anomaly_scores is not None
            else None
        )
        self.recon_errors = (
            torch.nan_to_num(recon_errors, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            if recon_errors is not None
            else None
        )
        self.names_map = names_map or {}
        self.types_map = types_map or {}

        # 预计算
        self._edge_importance_cache: Dict[int, float] = {}
    
    def set_intermediate_values(
        self,
        node_embeddings: torch.Tensor = None,
        attention_weights: torch.Tensor = None,
        cic_scores: torch.Tensor = None,
        cic_weights: torch.Tensor = None,
        anomaly_scores: torch.Tensor = None,
        recon_errors: torch.Tensor = None,
    ):
        """设置中间值（训练后调用）"""
        if node_embeddings is not None:
            self.node_embeddings = node_embeddings
        if attention_weights is not None:
            self.attention_weights = attention_weights
        if cic_scores is not None:
            self.cic_scores = cic_scores
        if cic_weights is not None:
            self.cic_weights = cic_weights
        if anomaly_scores is not None:
            self.anomaly_scores = torch.nan_to_num(anomaly_scores, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if recon_errors is not None:
            self.recon_errors = torch.nan_to_num(recon_errors, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        # 清除缓存
        self._edge_importance_cache.clear()

    @staticmethod
    def _cic_total_score(cic_scores: torch.Tensor, *, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute CIC total score in [0,1] consistent with utils.cic_invariants.InvariantScores.total_score().

        Accepts:
          - [n_nodes, 4] vector scores
          - [n_nodes] total scores
        """
        if cic_scores is None:
            raise ValueError("cic_scores is None")
        if cic_scores.dim() == 2:
            v = torch.nan_to_num(cic_scores, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            if weights is None:
                weights_t = torch.tensor([0.25, 0.25, 0.25, 0.25], device=v.device, dtype=v.dtype)
            else:
                weights_t = torch.as_tensor(weights, device=v.device, dtype=v.dtype).reshape(-1)
                if weights_t.numel() != 4:
                    raise ValueError(f"cic_weights must have 4 elements, got {weights_t.numel()}")
                weights_t = weights_t.clamp(0.0, 1.0)
            total = 1.0 - (1.0 - weights_t.view(1, 4) * v).prod(dim=-1)
            return total.clamp(0.0, 1.0)
        total = torch.nan_to_num(cic_scores, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return total.reshape(-1)

    def compute_edge_importance(self,
                                 edge_idx: int,
                                 src_id: int = None,
                                 dst_id: int = None) -> float:
        """
        计算边的重要性分数
        
        综合使用:
        - 注意力权重
        - 端点CIC分数
        - 端点重构误差
        """
        if edge_idx in self._edge_importance_cache:
            return self._edge_importance_cache[edge_idx]
        
        importance = 0.0
        weight_sum = 0.0
        
        # 1. 注意力权重
        if self.attention_weights is not None:
            attn = float(self.attention_weights[edge_idx].mean().item())
            attn = max(0.0, min(1.0, attn))
            importance += attn * 0.4
            weight_sum += 0.4

        # 2. 端点CIC分数
        if self.cic_scores is not None and src_id is not None and dst_id is not None:
            cic_total = self._cic_total_score(self.cic_scores, weights=self.cic_weights)
            src_cic = float(cic_total[src_id].item())
            dst_cic = float(cic_total[dst_id].item())
            cic_imp = (src_cic + dst_cic) / 2.0
            importance += cic_imp * 0.3
            weight_sum += 0.3

        # 3. 端点重构误差
        if self.recon_errors is not None and src_id is not None and dst_id is not None:
            src_err = float(self.recon_errors[src_id].item())
            dst_err = float(self.recon_errors[dst_id].item())
            err_imp = (src_err + dst_err) / 2.0
            importance += err_imp * 0.3
            weight_sum += 0.3

        if weight_sum > 0:
            importance = importance / weight_sum

        importance = max(0.0, min(1.0, float(importance)))
        self._edge_importance_cache[edge_idx] = importance
        return float(importance)

    def get_k_hop_neighbors(self,
                              center_node: int,
                              k: int = 2) -> Tuple[Set[int], Set[int]]:
        """
        获取k-hop邻居节点和边
        
        Returns:
            nodes: 邻居节点集合
            edges: 边索引集合
        """
        n_nodes = int(self.graph.num_nodes())
        if center_node < 0 or center_node >= n_nodes:
            raise ValueError(f"center_node out of range: {center_node} (num_nodes={n_nodes})")

        nodes: Set[int] = {int(center_node)}
        edge_ids: Set[int] = set()
        frontier: Set[int] = {int(center_node)}

        for _ in range(int(k)):
            new_frontier: Set[int] = set()
            for node in frontier:
                # out edges
                u, v, eids = self.graph.out_edges(node, form="all")
                for nid in v.tolist():
                    if nid not in nodes:
                        nodes.add(int(nid))
                        new_frontier.add(int(nid))
                for eid in eids.tolist():
                    edge_ids.add(int(eid))

                # in edges
                u, v, eids = self.graph.in_edges(node, form="all")
                for nid in u.tolist():
                    if nid not in nodes:
                        nodes.add(int(nid))
                        new_frontier.add(int(nid))
                for eid in eids.tolist():
                    edge_ids.add(int(eid))

            frontier = new_frontier
            if not frontier:
                break

        return nodes, edge_ids
    
    def build_subgraph(self,
                       center_node: int,
                       k_hop: int = 2,
                       threshold: float = 0.0,
                       anomaly_threshold: float = 0.5) -> ExplanationSubgraph:
        """
        构建以center_node为中心的k-hop解释子图
        
        Args:
            center_node: 中心节点ID
            k_hop: 扩展的跳数
            threshold: 边重要性阈值 (过滤低于阈值的边)
            
        Returns:
            ExplanationSubgraph对象
        """
        # 获取k-hop邻居
        node_ids, edge_idxs = self.get_k_hop_neighbors(center_node, k_hop)

        src, dst = self.graph.edges()

        # 构建节点列表
        nodes = []
        for nid in sorted(node_ids):
            nid = int(nid)
            display = self.names_map.get(nid, f"node_{nid}")
            node = ExplanationNode(
                node_id=nid,
                uuid=display,
                node_type=self.NODE_TYPE_NAMES.get(self.types_map.get(nid, 0), 'unknown'),
                name=display,
            )

            # 填充分数
            if self.anomaly_scores is not None:
                node.anomaly_score = float(self.anomaly_scores[nid].item())
                node.is_anomaly = node.anomaly_score > float(anomaly_threshold)

            if self.cic_scores is not None:
                if self.cic_scores.dim() == 2:
                    node.cic_scores = self.cic_scores[nid].cpu().tolist()
                else:
                    total = float(self.cic_scores[nid].item())
                    node.cic_scores = [total, 0.0, 0.0, 0.0]

            nodes.append(node)
        
        # 构建边列表
        edges = []
        for eidx in sorted(edge_idxs):
            eidx = int(eidx)
            src_id = int(src[eidx].item())
            dst_id = int(dst[eidx].item())

            importance = self.compute_edge_importance(eidx, src_id, dst_id)
            
            if importance < threshold:
                continue
            
            edge = ExplanationEdge(
                src_id=src_id,
                dst_id=dst_id,
                importance=importance,
            )
            
            # 填充注意力权重
            if self.attention_weights is not None:
                edge.attention_weight = float(self.attention_weights[eidx].mean().item())

            # 从图数据中获取边类型
            if 'type' in self.graph.edata:
                edge.edge_type = str(int(self.graph.edata['type'][eidx].item()))
            
            # 标记攻击边 (两端都是异常节点)
            if self.anomaly_scores is not None:
                src_anomaly = float(self.anomaly_scores[src_id].item()) > float(anomaly_threshold)
                dst_anomaly = float(self.anomaly_scores[dst_id].item()) > float(anomaly_threshold)
                edge.is_attack_edge = src_anomaly and dst_anomaly
            
            edges.append(edge)
        
        # 计算总异常分数
        total_score = 0.0
        if self.anomaly_scores is not None:
            for nid in node_ids:
                total_score += float(self.anomaly_scores[nid].item())
            total_score /= len(node_ids) if node_ids else 1.0
        
        return ExplanationSubgraph(
            center_node=center_node,
            nodes=nodes,
            edges=edges,
            k_hop=k_hop,
            total_anomaly_score=total_score
        )
    
    def find_attack_path(self, 
                          subgraph: ExplanationSubgraph,
                          start_node: Optional[int] = None,
                          end_node: Optional[int] = None) -> List[int]:
        """
        在子图中找到攻击路径
        
        基于边重要性使用最短路径变体
        """
        if not subgraph.edges:
            return []
        
        # 构建NetworkX图
        G = nx.DiGraph()
        for edge in subgraph.edges:
            # 权重 = 1 - importance (重要性高的边权重低)
            weight = 1.0 - edge.importance + 0.01
            G.add_edge(edge.src_id, edge.dst_id, weight=weight, importance=edge.importance)
        
        # 如果没有指定起点/终点，使用异常分数最高的节点
        if start_node is None or end_node is None:
            anomaly_nodes = [(n.node_id, n.anomaly_score) for n in subgraph.nodes if n.is_anomaly]
            if len(anomaly_nodes) >= 2:
                anomaly_nodes.sort(key=lambda x: x[1], reverse=True)
                if start_node is None:
                    start_node = anomaly_nodes[0][0]
                if end_node is None:
                    end_node = anomaly_nodes[-1][0]
            elif len(anomaly_nodes) == 1:
                start_node = end_node = anomaly_nodes[0][0]
            else:
                return []
        
        try:
            path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
            subgraph.attack_path = path
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_top_anomaly_nodes(self, k: int = 10) -> List[Tuple[int, float]]:
        """获取异常分数最高的k个节点"""
        if self.anomaly_scores is None:
            return []
        
        scores = self.anomaly_scores.cpu()
        top_k = min(k, scores.size(0))
        values, indices = torch.topk(scores, top_k)
        
        return [(int(idx.item()), float(val.item())) for idx, val in zip(indices, values)]
    
    def discover_communities(self, 
                              anomaly_threshold: float = 0.5) -> Dict[int, List[int]]:
        """
        使用社区发现算法定位异常社区
        
        类似attack_investigation.py中的Louvain算法
        """
        try:
            import community.community_louvain as community_louvain
        except ImportError:
            print("Warning: community_louvain not installed, skipping community discovery")
            return {}
        
        # 转换为NetworkX图
        nx_graph = self.graph.to_networkx().to_undirected()
        
        # 社区发现
        partition = community_louvain.best_partition(nx_graph)
        
        # 按社区分组
        communities: Dict[int, List[int]] = {}
        for node_id, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node_id)
        
        # 计算每个社区的异常分数
        if self.anomaly_scores is not None:
            comm_scores = {}
            for comm_id, nodes in communities.items():
                scores = [float(self.anomaly_scores[n].item()) for n in nodes]
                comm_scores[comm_id] = sum(scores) / len(scores) if scores else 0.0
            
            # 按异常分数排序
            sorted_comms = sorted(comm_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 只返回高于阈值的社区
            return {c: communities[c] for c, s in sorted_comms if s >= anomaly_threshold}
        
        return communities


# ============================================================================
# 可视化模块
# ============================================================================

class SubgraphVisualizer:
    """
    子图可视化器
    
    使用graphviz生成可视化PDF，类似attack_investigation.py
    """
    
    # 节点类型形状映射
    NODE_SHAPES = {
        'subject': 'box',
        'file': 'oval',
        'netflow': 'diamond',
        'memory': 'hexagon',
        'principal': 'pentagon',
        'unknown': 'circle',
    }
    
    def __init__(self, output_dir: str = './explanation_graphs'):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        s = "" if text is None else str(text)
        if len(s) <= max_len:
            return s
        if max_len <= 3:
            return s[:max_len]
        return s[: max_len - 3] + "..."

    @staticmethod
    def _abbrev_path(text: str) -> str:
        if not text:
            return text
        s = str(text)
        for prefix in ("/proc/", "/run/", "/tmp/", "/var/tmp/"):
            if prefix in s:
                s = s.replace(prefix, prefix.rstrip("/") + "/*/")
        if "/" in s:
            parts = [p for p in s.split("/") if p]
            if len(parts) >= 3:
                s = ".../" + "/".join(parts[-2:])
        return s

    @classmethod
    def _simplify_entity_label(cls, raw: str, *, max_len: int = 52) -> str:
        if raw is None:
            return ""
        s = str(raw)
        for key in ("subject", "file", "netflow", "memory", "principal"):
            token = f"'{key}': '"
            if token in s:
                start = s.find(token) + len(token)
                end = s.find("'", start)
                if end > start:
                    val = s[start:end]
                    if key == "file":
                        val = cls._abbrev_path(val)
                    return cls._truncate(val, max_len)
        s = cls._abbrev_path(s)
        return cls._truncate(s, max_len)

    @staticmethod
    def _safe_float(x: Any, *, default: float = 0.0) -> float:
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return default
            return v
        except Exception:
            return default

    def _prune_for_render(
        self,
        subgraph: ExplanationSubgraph,
        *,
        max_nodes: int = 200,
        max_edges: int = 400,
        keep_attack_path_only: bool = False,
        auto_edge_sigma: float = 1.5,
    ) -> ExplanationSubgraph:
        """Prune a potentially large explanation subgraph for visualization only."""
        if max_nodes <= 0 or max_edges <= 0:
            return subgraph

        if keep_attack_path_only and subgraph.attack_path and len(subgraph.attack_path) >= 2:
            keep_nodes = set(int(n) for n in subgraph.attack_path)
            keep_edges = set()
            for i in range(len(subgraph.attack_path) - 1):
                keep_edges.add((int(subgraph.attack_path[i]), int(subgraph.attack_path[i + 1])))
            nodes = [n for n in subgraph.nodes if int(n.node_id) in keep_nodes]
            edges = [e for e in subgraph.edges if (int(e.src_id), int(e.dst_id)) in keep_edges]
            return ExplanationSubgraph(
                center_node=subgraph.center_node,
                nodes=nodes,
                edges=edges,
                k_hop=subgraph.k_hop,
                community_id=subgraph.community_id,
                total_anomaly_score=subgraph.total_anomaly_score,
                attack_path=list(subgraph.attack_path),
            )

        threshold = None
        if subgraph.edges:
            imps = [self._safe_float(e.importance, default=0.0) for e in subgraph.edges]
            if len(imps) >= 2:
                threshold = statistics.mean(imps) + float(auto_edge_sigma) * statistics.pstdev(imps)

        attack_edges = set()
        if subgraph.attack_path and len(subgraph.attack_path) >= 2:
            for i in range(len(subgraph.attack_path) - 1):
                attack_edges.add((int(subgraph.attack_path[i]), int(subgraph.attack_path[i + 1])))

        kept_edges: List[ExplanationEdge] = []
        for e in subgraph.edges:
            pair = (int(e.src_id), int(e.dst_id))
            if pair in attack_edges:
                kept_edges.append(e)
                continue
            if threshold is not None and self._safe_float(e.importance, default=0.0) >= float(threshold):
                kept_edges.append(e)

        if len(kept_edges) > max_edges:
            kept_edges.sort(key=lambda ee: self._safe_float(ee.importance, default=0.0), reverse=True)
            kept_edges = kept_edges[:max_edges]

        if len(kept_edges) < min(30, max_edges) and subgraph.edges:
            remaining = [e for e in subgraph.edges if e not in kept_edges]
            remaining.sort(key=lambda ee: self._safe_float(ee.importance, default=0.0), reverse=True)
            need = min(max_edges - len(kept_edges), max(0, min(50, max_edges) - len(kept_edges)))
            kept_edges.extend(remaining[:need])

        keep_node_ids = {int(subgraph.center_node)}
        for e in kept_edges:
            keep_node_ids.add(int(e.src_id))
            keep_node_ids.add(int(e.dst_id))

        node_by_id = {int(n.node_id): n for n in subgraph.nodes}
        kept_nodes = [node_by_id[nid] for nid in keep_node_ids if nid in node_by_id]

        if len(kept_nodes) > max_nodes:
            priority: List[int] = []
            center = int(subgraph.center_node)
            if center in node_by_id:
                priority.append(center)
            if subgraph.attack_path:
                for nid in subgraph.attack_path:
                    nid = int(nid)
                    if nid in node_by_id and nid not in priority:
                        priority.append(nid)
            for n in kept_nodes:
                if n.is_anomaly and int(n.node_id) not in priority:
                    priority.append(int(n.node_id))
            rest = [n for n in kept_nodes if int(n.node_id) not in priority]
            rest.sort(key=lambda nn: float(nn.anomaly_score), reverse=True)
            for n in rest:
                if len(priority) >= max_nodes:
                    break
                priority.append(int(n.node_id))

            keep_node_ids = set(priority[:max_nodes])
            kept_nodes = [node_by_id[nid] for nid in priority[:max_nodes] if nid in node_by_id]
            kept_edges = [e for e in kept_edges if int(e.src_id) in keep_node_ids and int(e.dst_id) in keep_node_ids]

        return ExplanationSubgraph(
            center_node=subgraph.center_node,
            nodes=kept_nodes,
            edges=kept_edges,
            k_hop=subgraph.k_hop,
            community_id=subgraph.community_id,
            total_anomaly_score=subgraph.total_anomaly_score,
            attack_path=list(subgraph.attack_path),
        )
    
    # 论文级别导出格式列表
    PAPER_FORMATS = ['pdf', 'eps', 'svg', 'png', 'tiff', 'jpeg']
    DEFAULT_DPI = 1200
    
    def visualize_subgraph(self,
                            subgraph: ExplanationSubgraph,
                            filename: str = 'explanation',
                            format: str = 'pdf',
                            show_scores: bool = True,
                            highlight_attack_path: bool = True,
                            *,
                            style: str = "paper",
                            show_legend: bool = True,
                            max_nodes: int = 200,
                            max_edges: int = 400,
                            keep_attack_path_only: bool = False,
                            export_all_formats: bool = False,
                            dpi: int = 1200) -> str:
        """
        可视化解释子图
        
        Args:
            subgraph: ExplanationSubgraph对象
            filename: 输出文件名
            format: 输出格式 ('pdf', 'png', 'svg', 'eps', 'tiff', 'jpeg')
            show_scores: 是否显示分数
            highlight_attack_path: 是否高亮攻击路径
            export_all_formats: 是否导出所有论文格式 (PDF, EPS, SVG, PNG, TIFF, JPEG)
            dpi: 图像DPI (默认1200，论文级别)
            
        Returns:
            输出文件路径 (主格式)
        """
        try:
            from graphviz import Digraph
        except ImportError:
            print("Warning: graphviz not installed, cannot visualize")
            return ""
        
        render_sg = self._prune_for_render(
            subgraph,
            max_nodes=max_nodes,
            max_edges=max_edges,
            keep_attack_path_only=keep_attack_path_only,
        )

        dot = Digraph(name="ExplanationSubgraph", format=format)
        dot.graph_attr.update({
            'rankdir': 'LR',
            'dpi': str(dpi),  # 论文级别 DPI
            'splines': 'spline',
            'nodesep': '0.30',
            'ranksep': '0.35',
            'fontname': 'Helvetica',
            'fontsize': '10',
        })
        dot.node_attr.update({'fontname': 'Helvetica', 'fontsize': '10', 'margin': '0.06,0.04'})
        dot.edge_attr.update({'fontname': 'Helvetica', 'fontsize': '9'})
        
        # 创建节点ID到ExplanationNode的映射
        node_map = {n.node_id: n for n in render_sg.nodes}
        
        # 攻击路径边集合
        attack_path_edges = set()
        if highlight_attack_path and render_sg.attack_path:
            for i in range(len(render_sg.attack_path) - 1):
                attack_path_edges.add((int(render_sg.attack_path[i]), int(render_sg.attack_path[i+1])))

        # Community id on the rendered subgraph (optional; improves readability like attack_investigation.py)
        partition: Dict[int, int] = {}
        if style in {"paper", "attack_investigation"}:
            try:
                import community.community_louvain as community_louvain

                gnx = nx.DiGraph()
                for e in render_sg.edges:
                    gnx.add_edge(int(e.src_id), int(e.dst_id))
                if gnx.number_of_nodes() > 0 and gnx.number_of_edges() > 0:
                    partition = community_louvain.best_partition(gnx.to_undirected())
            except Exception:
                partition = {}
        
        # 添加节点
        for node in render_sg.nodes:
            shape = self.NODE_SHAPES.get(node.node_type, 'circle')

            is_center = int(node.node_id) == int(render_sg.center_node)
            comm_id = partition.get(int(node.node_id))

            short_name = self._simplify_entity_label(node.name, max_len=52)
            if style == "attack_investigation":
                label = short_name + (f" C{comm_id}" if comm_id is not None else "")
            else:
                label = f"{node.node_type or 'unknown'}: {short_name}"
                if comm_id is not None:
                    label += f"\nC{comm_id}"
                if show_scores:
                    label += f"\nscore={float(node.anomaly_score):.2f}"

            if style == "attack_investigation":
                color = 'red' if node.is_anomaly else 'blue'
                dot.node(
                    name=str(node.node_id),
                    label=label,
                    shape=shape,
                    color=('purple4' if is_center else color),
                    penwidth='2.2' if is_center else '1.2',
                    style='solid'
                )
            else:
                border = '#b00020' if node.is_anomaly else '#2457c5'
                fill = '#ffe6e6' if node.is_anomaly else '#ffffff'
                if node.node_type == 'subject' and not node.is_anomaly:
                    fill = '#e8f1ff'
                elif node.node_type == 'file' and not node.is_anomaly:
                    fill = '#e8ffe8'
                elif node.node_type == 'netflow' and not node.is_anomaly:
                    fill = '#fff3e0'
                elif node.node_type == 'memory' and not node.is_anomaly:
                    fill = '#f2f2f2'
                if is_center:
                    border = '#6a1bb1'
                    fill = '#f2e6ff'
                dot.node(
                    name=str(node.node_id),
                    label=label,
                    shape=shape,
                    color=border,
                    style='filled',
                    fillcolor=fill,
                    penwidth='2.4' if is_center else ('1.8' if node.is_anomaly else '1.2')
                )
        
        # 添加边
        for edge in render_sg.edges:
            on_path = (int(edge.src_id), int(edge.dst_id)) in attack_path_edges
            if style == "attack_investigation":
                color = 'red' if (edge.is_attack_edge or on_path) else 'blue'
                penwidth = '2.0' if on_path else ('1.6' if edge.is_attack_edge else '1.0')
            else:
                if on_path:
                    color = '#b00020'
                    penwidth = '3.0'
                elif edge.is_attack_edge:
                    color = '#f57c00'
                    penwidth = '2.0'
                else:
                    color = '#9aa0a6'
                    penwidth = '1.0'

            label = f"{edge.edge_type}" if edge.edge_type else ""
            if show_scores and style != "attack_investigation":
                label += (("\n" if label else "") + f"imp={float(edge.importance):.2f}")

            dot.edge(
                str(edge.src_id),
                str(edge.dst_id),
                label=label,
                color=color,
                penwidth=penwidth
            )

        if show_legend and style != "attack_investigation":
            with dot.subgraph(name='cluster_legend') as c:
                c.attr(label='Legend', color='#d0d0d0', fontname='Helvetica', fontsize='10')
                c.node('L_subject', label='subject', shape='box', style='filled', fillcolor='#e8f1ff', color='#2457c5')
                c.node('L_file', label='file', shape='oval', style='filled', fillcolor='#e8ffe8', color='#2457c5')
                c.node('L_netflow', label='netflow', shape='diamond', style='filled', fillcolor='#fff3e0', color='#2457c5')
                c.node('L_anom', label='anomalous', shape='circle', style='filled', fillcolor='#ffe6e6', color='#b00020')
                c.edge('L_subject', 'L_file', label='edge', color='#9aa0a6', penwidth='1.0')
                c.edge('L_file', 'L_netflow', label='attack path', color='#b00020', penwidth='3.0')
        
        # 渲染主格式
        output_path = os.path.join(self.output_dir, filename)
        dot.render(output_path, view=False, cleanup=True)
        primary_output = f"{output_path}.{format}"
        
        # 如果需要导出所有论文格式
        if export_all_formats:
            exported_files = [primary_output]
            for fmt in self.PAPER_FORMATS:
                if fmt == format:
                    continue  # 跳过已经导出的主格式
                try:
                    dot.format = fmt
                    dot.graph_attr['dpi'] = str(dpi)
                    dot.render(output_path, view=False, cleanup=True)
                    exported_files.append(f"{output_path}.{fmt}")
                except Exception as e:
                    print(f"[WARN] Failed to export {fmt}: {e}")
            print(f"[OK] Exported {len(exported_files)} formats: {', '.join(f.split('.')[-1] for f in exported_files)}")
        
        return primary_output
    
    def visualize_attack_summary(self,
                                  subgraphs: List[ExplanationSubgraph],
                                  filename: str = 'attack_summary',
                                  format: str = 'jpeg',
                                  export_all_formats: bool = False,
                                  dpi: int = 1200) -> str:
        """
        生成攻击汇总可视化
        
        显示所有发现的异常子图概览
        
        Args:
            subgraphs: 子图列表
            filename: 输出文件名
            format: 输出格式 ('pdf', 'png', 'svg', 'eps', 'tiff', 'jpeg')
            export_all_formats: 是否导出所有论文格式
            dpi: 图像DPI (默认1200)
        """
        try:
            from graphviz import Digraph
        except ImportError:
            return ""
        
        dot = Digraph(name="AttackSummary", format=format)
        dot.graph_attr['rankdir'] = 'TB'
        dot.graph_attr['dpi'] = str(dpi)
        
        for i, sg in enumerate(subgraphs):
            with dot.subgraph(name=f'cluster_{i}') as c:
                c.attr(label=f'Subgraph {i} (score: {sg.total_anomaly_score:.2f})')
                c.attr(color='red' if sg.total_anomaly_score > 0.5 else 'blue')
                
                for node in sg.nodes[:10]:  # 限制显示数量
                    c.node(f'{i}_{node.node_id}', 
                           label=node.name[:20],
                           color='red' if node.is_anomaly else 'blue')
        
        output_path = os.path.join(self.output_dir, filename)
        dot.render(output_path, view=False, cleanup=True)
        primary_output = f"{output_path}.{format}"
        
        # 如果需要导出所有论文格式
        if export_all_formats:
            exported_files = [primary_output]
            for fmt in self.PAPER_FORMATS:
                if fmt == format:
                    continue
                try:
                    dot.format = fmt
                    dot.graph_attr['dpi'] = str(dpi)
                    dot.render(output_path, view=False, cleanup=True)
                    exported_files.append(f"{output_path}.{fmt}")
                except Exception as e:
                    print(f"[WARN] Failed to export {fmt}: {e}")
            print(f"[OK] Exported {len(exported_files)} formats for summary")
        
        return primary_output


# ============================================================================
# 中间值保存/加载
# ============================================================================

def save_intermediate_values(save_dir: str,
                              epoch: int,
                              attention_weights: torch.Tensor = None,
                              node_embeddings: torch.Tensor = None,
                              cic_scores: torch.Tensor = None,
                              fusion_scores: torch.Tensor = None,
                              recon_errors: torch.Tensor = None):
    """
    保存训练过程中的中间值
    
    按照implementation_plan.md的要求保存
    """
    os.makedirs(save_dir, exist_ok=True)
    
    values = {}
    
    if attention_weights is not None:
        values['attention_weights'] = attention_weights.detach().cpu()
    
    if node_embeddings is not None:
        values['node_embeddings'] = node_embeddings.detach().cpu()
    
    if cic_scores is not None:
        values['cic_scores'] = cic_scores.detach().cpu()
    
    if fusion_scores is not None:
        values['fusion_scores'] = fusion_scores.detach().cpu()
    
    if recon_errors is not None:
        values['recon_errors'] = recon_errors.detach().cpu()
    
    save_path = os.path.join(save_dir, f'intermediate_epoch_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pkl.dump(values, f)
    
    print(f"[Explanation] 已保存中间值到 {save_path}")
    return save_path


def load_intermediate_values(save_dir: str, epoch: int) -> Dict[str, torch.Tensor]:
    """加载中间值"""
    load_path = os.path.join(save_dir, f'intermediate_epoch_{epoch}.pkl')
    
    if not os.path.exists(load_path):
        return {}
    
    with open(load_path, 'rb') as f:
        values = pkl.load(f)
    
    print(f"[Explanation] 已加载中间值从 {load_path}")
    return values


# ============================================================================
# 与模型模块对接的辅助函数
# ============================================================================

@torch.no_grad()
def extract_gat_attention_weights(
    encoder: torch.nn.Module,
    g: dgl.DGLGraph,
    node_features: torch.Tensor,
    *,
    layer: int = -1,
) -> Optional[torch.Tensor]:
    """
    Extract edge attention weights from `model/gat.py` GAT encoder.

    Returns:
        attention: [n_edges, n_heads, 1] (or similar) on the same device as `g`, or None if not supported.
    """
    if encoder is None or not hasattr(encoder, "gats"):
        return None
    gats = getattr(encoder, "gats")
    if not isinstance(gats, torch.nn.ModuleList):
        return None

    n_layers = len(gats)
    if n_layers == 0:
        return None
    chosen = n_layers - 1 if layer == -1 else int(layer)
    if chosen < 0 or chosen >= n_layers:
        raise ValueError(f"layer out of range: {layer} (n_layers={n_layers})")

    h = node_features
    attn = None
    for i, conv in enumerate(gats):
        if i == chosen:
            try:
                h, attn = conv(g, h, get_attention=True)
            except TypeError:
                return None
        else:
            h = conv(g, h)
    return attn


@torch.no_grad()
def prepare_explanation_builder_from_modules(
    g: dgl.DGLGraph,
    *,
    model: Optional[torch.nn.Module] = None,
    cic_scores: Optional[torch.Tensor] = None,
    anomaly_scorer: Optional[torch.nn.Module] = None,
    node_contrastive: Optional[torch.nn.Module] = None,
    names_map: Optional[Dict[int, str]] = None,
    types_map: Optional[Dict[int, int]] = None,
    anomaly_threshold: float = 0.5,
) -> ExplanationSubgraphBuilder:
    """
    Convenience factory that wires CIC + reconstruction + contrastive scores into ExplanationSubgraphBuilder.

    - `cic_scores` should be [n_nodes, 4] from `utils/loaddata.transform_graph_with_cic` (g.ndata["cic_scores"]).
    - `model` can be `model/autoencoder.py:GMAEModel` which provides `embed()` and `node_reconstruction_error()`.
    - `node_contrastive` can be `model/contrastive.py:NodeLevelContrastive` which provides `anomaly_score()`.
    - `anomaly_scorer` can be `model/fusion.py:AnomalyScorer`.
    """
    node_embeddings = None
    recon_errors = None
    attention_weights = None
    contrastive_score = None
    fused_scores = None

    if model is not None and hasattr(model, "embed"):
        node_embeddings = model.embed(g)
        if isinstance(node_embeddings, tuple):
            node_embeddings = node_embeddings[0]

    if model is not None and hasattr(model, "node_reconstruction_error"):
        recon_errors = model.node_reconstruction_error(g)

    if model is not None and hasattr(model, "encoder") and "attr" in g.ndata:
        try:
            attention_weights = extract_gat_attention_weights(model.encoder, g, g.ndata["attr"], layer=-1)
        except Exception:
            attention_weights = None

    if node_contrastive is not None and node_embeddings is not None and cic_scores is not None:
        if hasattr(node_contrastive, "anomaly_score"):
            contrastive_score = node_contrastive.anomaly_score(node_embeddings, cic_scores, threshold=anomaly_threshold)

    if anomaly_scorer is not None and cic_scores is not None:
        # anomaly_scorer can accept node-level scores from contrastive/reconstruction to produce fused node scores.
        if hasattr(anomaly_scorer, "compute_anomaly_score"):
            fused_scores = anomaly_scorer.compute_anomaly_score(
                cic_scores, contrastive_score=contrastive_score, recon_error=recon_errors
            )
        else:
            try:
                out = anomaly_scorer(cic_scores, contrastive_score=contrastive_score, recon_error=recon_errors)
                fused_scores = out.get("anomaly_score") if isinstance(out, dict) else out
            except Exception:
                fused_scores = None

    return ExplanationSubgraphBuilder(
        g,
        node_embeddings=node_embeddings,
        attention_weights=attention_weights,
        cic_scores=cic_scores,
        cic_weights=(getattr(getattr(anomaly_scorer, "invariant_fusion", None), "weights", None)
                     if anomaly_scorer is not None else None),
        anomaly_scores=fused_scores,
        recon_errors=recon_errors,
        names_map=names_map,
        types_map=types_map,
    )


# ============================================================================
# 工厂函数
# ============================================================================

def create_explanation_builder(graph: dgl.DGLGraph,
                                **kwargs) -> ExplanationSubgraphBuilder:
    """创建解释子图构建器"""
    return ExplanationSubgraphBuilder(graph, **kwargs)


def create_visualizer(output_dir: str = './explanation_graphs') -> SubgraphVisualizer:
    """创建可视化器"""
    return SubgraphVisualizer(output_dir)
