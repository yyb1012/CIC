"""
负样本构建模块 (Negative Sample Construction)

基于CIC不变量违反构建对比学习的负样本：
- 身份错配: 替换src_uid, src_gid, src_exe_hash
- 跨命名空间: 修改mnt_ns, pid_ns, net_ns
- 时序逆序: 交换相邻事件timestamp
- 权限提升: 替换euid, egid为更高权限
- 路径错位: 浏览器进程写入系统敏感路径

参考: FIELD_DOCUMENTATION.md 第二节
"""

import torch
import torch.nn as nn
import random
from typing import Dict, List, Optional, Tuple, Any
import dgl


def _pick_indices(n: int, k: int, device: torch.device) -> torch.Tensor:
    """
    Pick k indices from [0, n) on `device`.

    For very large n, torch.randperm(n) is expensive (O(n) memory/time), so we use randint with replacement.
    """
    if n <= 0 or k <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if k >= n:
        return torch.arange(n, device=device)
    return torch.randint(0, n, (k,), device=device)


def _pick_unique_indices(n: int, k: int, device: torch.device) -> torch.Tensor:
    if n <= 0 or k <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if k >= n:
        return torch.arange(n, device=device)
    return torch.randperm(n, device=device)[:k]


class NegativeSampleBuilder:
    """
    基于不变量违反构建负样本
    
    通过人工破坏CIC不变量来构造负样本，
    使模型学会区分正常模式和异常模式。
    """
    
    def __init__(self, 
                 violation_types: Optional[List[str]] = None,
                 perturbation_rate: float = 0.3):
        """
        初始化负样本构建器
        
        Args:
            violation_types: 要使用的违反类型列表
                - 'identity': 身份错配
                - 'namespace': 跨命名空间
                - 'timing': 时序逆序
                - 'privilege': 权限提升
                - 'path': 路径错位
            perturbation_rate: 每种类型中扰动的节点/边比例
        """
        if violation_types is None:
            violation_types = ['identity', 'namespace', 'timing', 'privilege', 'path']
        self.violation_types = violation_types
        self.perturbation_rate = perturbation_rate
    
    def build_identity_mismatch(self, 
                                 g: dgl.DGLGraph,
                                 node_attrs: Dict[str, torch.Tensor]) -> Tuple[dgl.DGLGraph, Dict[str, torch.Tensor]]:
        """
        身份错配: 替换部分节点的uid, gid, exe_hash
        
        破坏 I_creator 不变量
        """
        new_g = g.clone()
        new_attrs = {k: v.clone() for k, v in node_attrs.items()}
        
        n_nodes = g.num_nodes()
        if n_nodes == 0:
            return new_g, new_attrs
        num_perturb = max(1, int(n_nodes * self.perturbation_rate))
        device = next(iter(new_attrs.values())).device if new_attrs else g.device
        perturb_nodes = _pick_indices(n_nodes, num_perturb, device=device)
        
        # 对于每个被扰动的节点，随机替换其身份属性
        for attr_name in ['uid', 'gid', 'exe_hash']:
            if attr_name in new_attrs:
                attr_tensor = new_attrs[attr_name]
                donor_nodes = _pick_indices(n_nodes, num_perturb, device=attr_tensor.device)
                attr_tensor[perturb_nodes] = attr_tensor[donor_nodes]
                new_g.ndata[attr_name] = attr_tensor
        
        return new_g, new_attrs
    
    def build_namespace_violation(self,
                                   g: dgl.DGLGraph,
                                   node_attrs: Dict[str, torch.Tensor]) -> Tuple[dgl.DGLGraph, Dict[str, torch.Tensor]]:
        """
        跨命名空间: 修改mnt_ns, pid_ns使其与邻接节点不一致
        
        破坏 I_reach 不变量
        """
        new_g = g.clone()
        new_attrs = {k: v.clone() for k, v in node_attrs.items()}
        
        n_nodes = g.num_nodes()
        if n_nodes == 0:
            return new_g, new_attrs
        num_perturb = max(1, int(n_nodes * self.perturbation_rate))
        device = next(iter(new_attrs.values())).device if new_attrs else g.device
        perturb_nodes = _pick_indices(n_nodes, num_perturb, device=device)
        
        for attr_name in ['mnt_ns', 'pid_ns', 'net_ns']:
            if attr_name in new_attrs:
                attr_tensor = new_attrs[attr_name]
                # 使用一个不太可能出现的特殊值
                if attr_tensor.dtype in [torch.float32, torch.float64]:
                    attr_tensor[perturb_nodes] = -999.0
                else:
                    attr_tensor[perturb_nodes] = -999
                new_g.ndata[attr_name] = attr_tensor
        
        return new_g, new_attrs
    
    def build_timing_reversal(self,
                               g: dgl.DGLGraph,
                               edge_attrs: Dict[str, torch.Tensor]) -> Tuple[dgl.DGLGraph, Dict[str, torch.Tensor]]:
        """
        时序逆序: 交换相邻边的timestamp
        
        破坏 I_timing 不变量
        """
        new_g = g.clone()
        new_attrs = {k: v.clone() for k, v in edge_attrs.items()}
        
        if 'timestamp' not in new_attrs:
            return new_g, new_attrs
        
        n_edges = g.num_edges()
        if n_edges < 2:
            return new_g, new_attrs
        
        timestamps = new_attrs['timestamp']
        
        # 选择要交换的边对
        num_swaps = max(1, int(n_edges * self.perturbation_rate / 2))
        
        for _ in range(num_swaps):
            i, j = random.sample(range(n_edges), 2)
            # 交换时间戳
            timestamps[i], timestamps[j] = timestamps[j].clone(), timestamps[i].clone()

        new_g.edata['timestamp'] = timestamps
        
        return new_g, new_attrs
    
    def build_privilege_escalation(self,
                                     g: dgl.DGLGraph,
                                     node_attrs: Dict[str, torch.Tensor]) -> Tuple[dgl.DGLGraph, Dict[str, torch.Tensor]]:
        """
        权限提升: 将euid/egid替换为更高权限 (如0=root)
        
        破坏权限相关的安全约束
        """
        new_g = g.clone()
        new_attrs = {k: v.clone() for k, v in node_attrs.items()}
        
        n_nodes = g.num_nodes()
        if n_nodes == 0:
            return new_g, new_attrs
        num_perturb = max(1, int(n_nodes * self.perturbation_rate))
        device = next(iter(new_attrs.values())).device if new_attrs else g.device
        perturb_nodes = _pick_indices(n_nodes, num_perturb, device=device)
        
        for attr_name in ['euid', 'egid']:
            if attr_name in new_attrs:
                attr_tensor = new_attrs[attr_name]
                # 设置为root权限 (0)
                if attr_tensor.dtype in [torch.float32, torch.float64]:
                    attr_tensor[perturb_nodes] = 0.0
                else:
                    attr_tensor[perturb_nodes] = 0
                new_g.ndata[attr_name] = attr_tensor
        
        return new_g, new_attrs
    
    def build_path_mismatch(self,
                              g: dgl.DGLGraph,
                              node_types: torch.Tensor,
                              edge_attrs: Dict[str, torch.Tensor],
                              sensitive_paths: Optional[List[str]] = None) -> Tuple[dgl.DGLGraph, Dict[str, torch.Tensor]]:
        """
        路径错位: 模拟浏览器进程写入系统敏感路径等异常模式
        
        通过混淆边的目标来破坏正常的访问模式
        """
        new_attrs = {k: v.clone() for k, v in edge_attrs.items()}
        
        if sensitive_paths is None:
            sensitive_paths = ['/etc/', '/usr/bin/', '/root/', '/var/log/']
        
        # 这里我们通过交换边的目标来模拟路径错位
        n_edges = g.num_edges()
        if n_edges < 2:
            return g.clone(), new_attrs
        
        # 获取边信息
        src, dst = g.edges()
        
        num_perturb = max(1, int(n_edges * self.perturbation_rate))
        perturb_edges = _pick_unique_indices(n_edges, num_perturb, device=dst.device)
        
        # 对于选中的边，将其目标节点与其他边的目标交换
        new_dst = dst.clone()
        for idx in perturb_edges:
            swap_with = random.randint(0, n_edges - 1)
            new_dst[idx] = dst[swap_with]

        # 重建图（保持边数/顺序一致），以真正产生结构负样本
        new_g = dgl.graph((src, new_dst), num_nodes=g.num_nodes(), device=g.device)
        for key, value in g.ndata.items():
            new_g.ndata[key] = value.clone()
        for key, value in g.edata.items():
            new_g.edata[key] = value.clone()

        if 'perturbed' not in new_attrs:
            new_attrs['perturbed'] = torch.zeros(n_edges, dtype=torch.bool, device=g.device)
        new_attrs['perturbed'][perturb_edges] = True
        new_g.edata['perturbed'] = new_attrs['perturbed']

        return new_g, new_attrs
    
    def build_negative_sample(self,
                               g: dgl.DGLGraph,
                               violation_type: str,
                               node_attrs: Optional[Dict[str, torch.Tensor]] = None,
                               edge_attrs: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[dgl.DGLGraph, str]:
        """
        构建单个负样本
        
        Args:
            g: 原始图
            violation_type: 违反类型
            node_attrs: 节点属性字典
            edge_attrs: 边属性字典
            
        Returns:
            negative_g: 负样本图
            violation_type: 违反类型标签
        """
        if node_attrs is None:
            node_attrs = dict(g.ndata)
        if edge_attrs is None:
            edge_attrs = dict(g.edata)
        
        if violation_type == 'identity':
            neg_g, _ = self.build_identity_mismatch(g, node_attrs)
        elif violation_type == 'namespace':
            neg_g, _ = self.build_namespace_violation(g, node_attrs)
        elif violation_type == 'timing':
            neg_g, _ = self.build_timing_reversal(g, edge_attrs)
        elif violation_type == 'privilege':
            neg_g, _ = self.build_privilege_escalation(g, node_attrs)
        elif violation_type == 'path':
            node_types = g.ndata.get('type', torch.zeros(g.num_nodes()))
            neg_g, _ = self.build_path_mismatch(g, node_types, edge_attrs)
        else:
            raise ValueError(f"Unknown violation type: {violation_type}")
        
        return neg_g, violation_type
    
    def build_all_negatives(self,
                             g: dgl.DGLGraph,
                             node_attrs: Optional[Dict[str, torch.Tensor]] = None,
                             edge_attrs: Optional[Dict[str, torch.Tensor]] = None) -> List[Tuple[dgl.DGLGraph, str]]:
        """
        构建所有类型的负样本
        
        Returns:
            List of (negative_graph, violation_type) tuples
        """
        negatives = []
        for vtype in self.violation_types:
            try:
                neg_g, label = self.build_negative_sample(g, vtype, node_attrs, edge_attrs)
                negatives.append((neg_g, label))
            except Exception as e:
                print(f"Warning: Failed to build {vtype} negative: {e}")
        
        return negatives


class FeaturePerturbation(nn.Module):
    """
    特征级扰动模块
    
    直接在特征空间中构建负样本，适用于one-hot编码的特征。
    """
    
    def __init__(self, 
                 perturbation_rate: float = 0.2,
                 perturbation_strength: float = 0.5):
        """
        初始化特征扰动模块
        
        Args:
            perturbation_rate: 被扰动的节点/边比例
            perturbation_strength: 扰动强度
        """
        super().__init__()
        self.perturbation_rate = perturbation_rate
        self.perturbation_strength = perturbation_strength
    
    def forward(self, 
                features: torch.Tensor,
                cic_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        对特征进行扰动
        
        Args:
            features: 原始特征 [n, d]
            cic_scores: CIC分数 [n, 4] (可选，用于指导扰动)
            
        Returns:
            perturbed_features: 扰动后的特征
        """
        n = features.size(0)
        if n == 0:
            return features
        num_perturb = max(1, int(n * self.perturbation_rate))
        device = features.device
        
        # 选择要扰动的节点
        if cic_scores is not None:
            # 优先扰动低分数节点（使其看起来更像高分数节点）
            if cic_scores.dim() == 2:
                scores = cic_scores.sum(dim=1)
            else:
                scores = cic_scores
            _, indices = torch.topk(-scores, num_perturb)  # 取最低分
        else:
            indices = _pick_indices(n, num_perturb, device=device)
        
        # 扰动方式: 添加噪声或混合其他特征
        perturbed = features.clone()
        
        # 高斯噪声
        noise = torch.randn_like(perturbed[indices]) * self.perturbation_strength
        perturbed[indices] = perturbed[indices] + noise
        
        # 混合其他节点的特征
        donor_indices = _pick_indices(n, num_perturb, device=device)
        mix_weight = float(max(0.0, min(1.0, self.perturbation_strength)))
        perturbed[indices] = (1 - mix_weight) * perturbed[indices] + mix_weight * features[donor_indices]
        
        return perturbed


class EdgeDropSampler:
    """
    边级负样本采样器
    
    通过随机删除边来构建负样本，测试结构重构能力。
    """
    
    def __init__(self, drop_rate: float = 0.2):
        """
        Args:
            drop_rate: 删除边的比例
        """
        self.drop_rate = drop_rate
    
    def sample(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        采样边删除的负样本
        
        Args:
            g: 原始图
            
        Returns:
            sampled_g: 删除了部分边的图
        """
        n_edges = g.num_edges()
        if n_edges == 0:
            return g.clone()
        num_keep = max(1, int(n_edges * (1 - self.drop_rate)))
        
        # 随机选择要保留的边
        keep_edges = _pick_unique_indices(n_edges, num_keep, device=g.device)
        
        # 创建子图
        src, dst = g.edges()
        new_src = src[keep_edges]
        new_dst = dst[keep_edges]
        
        new_g = dgl.graph((new_src, new_dst), num_nodes=g.num_nodes(), device=g.device)
        
        # 复制节点属性
        for key, value in g.ndata.items():
            new_g.ndata[key] = value.clone()
        
        # 复制边属性 (只保留选中的边)
        for key, value in g.edata.items():
            new_g.edata[key] = value[keep_edges].clone()
        
        return new_g


# ============================================================================
# 工厂函数
# ============================================================================

def create_negative_builder(violation_types: Optional[List[str]] = None,
                             perturbation_rate: float = 0.3) -> NegativeSampleBuilder:
    """创建负样本构建器"""
    return NegativeSampleBuilder(
        violation_types=violation_types,
        perturbation_rate=perturbation_rate
    )
