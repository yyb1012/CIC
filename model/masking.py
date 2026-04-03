"""
不变量感知掩码模块 (Invariant-Aware Masking)

基于CIC不变量分数实现智能掩码策略：
- 优先掩码高违例分数的节点
- 基于因果链的掩码
- 基于属性类型的差异化掩码

参考: FIELD_DOCUMENTATION.md 第二节
"""

import torch
import torch.nn as nn
import random
import math
from typing import Dict, List, Optional, Tuple, Any
import dgl


class InvariantAwareMasking(nn.Module):
    """
    基于CIC不变量的智能掩码策略
    
    相比随机掩码，该模块会考虑节点的不变量违例分数，
    优先掩码那些可能更"有信息量"的节点。
    """
    
    def __init__(self, 
                 mask_rate: float = 0.5,
                 strategy: str = 'hybrid',
                 violation_weight: float = 0.3,
                 random_weight: float = 0.7):
        """
        初始化掩码模块
        
        Args:
            mask_rate: 掩码比例 [0, 1]
            strategy: 掩码策略
                - 'random': 纯随机掩码
                - 'violation': 基于违例分数的掩码 (高分优先)
                - 'inverse_violation': 反向违例掩码 (低分优先，用于对比)
                - 'hybrid': 混合策略 (默认)
            violation_weight: 违例分数掩码的权重 (仅hybrid策略)
            random_weight: 随机掩码的权重 (仅hybrid策略)
        """
        super().__init__()
        if not (0.0 <= mask_rate <= 1.0):
            raise ValueError(f"mask_rate must be in [0, 1], got {mask_rate}")
        self.mask_rate = mask_rate
        self.strategy = strategy
        if violation_weight < 0 or random_weight < 0:
            raise ValueError("violation_weight/random_weight must be non-negative")
        self.violation_weight = violation_weight
        self.random_weight = random_weight
        
        # 可学习的掩码token
        self.mask_token = nn.Parameter(torch.zeros(1, 1))  # 将在forward时扩展
    
    def init_mask_token(self, feature_dim: int):
        """初始化掩码token维度"""
        self.mask_token = nn.Parameter(torch.zeros(1, feature_dim))
        nn.init.normal_(self.mask_token, std=0.02)
    
    def compute_mask_probs(self, 
                           g: dgl.DGLGraph, 
                           cic_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算每个节点被掩码的概率
        
        Args:
            g: DGL图
            cic_scores: CIC分数张量 [n_nodes, 4] 或 [n_nodes] (总分)
            
        Returns:
            mask_probs: 掩码概率 [n_nodes]
        """
        n_nodes = g.num_nodes()
        device = g.device
        if n_nodes == 0:
            return torch.empty(0, device=device)
        
        if self.strategy == 'random' or cic_scores is None:
            # 均匀概率
            return torch.ones(n_nodes, device=device) / n_nodes
        
        # 获取总违例分数
        if cic_scores.dim() == 2:
            # 如果是4维向量，计算总分 (风险放大聚合)
            total_scores = self._compute_total_score(cic_scores)
        else:
            total_scores = cic_scores

        if total_scores.numel() != n_nodes:
            raise ValueError(f"cic_scores length mismatch: got {total_scores.numel()}, expected {n_nodes}")
        total_scores = total_scores.to(device=device, dtype=torch.float32)
        total_scores = total_scores.clamp(0.0, 1.0)
        
        if self.strategy == 'violation':
            # 高违例分数 -> 高掩码概率
            probs = total_scores + 1e-6  # 避免全0
        elif self.strategy == 'inverse_violation':
            # 低违例分数 -> 高掩码概率 (掩码正常节点，测试异常检测能力)
            probs = 1.0 - total_scores + 1e-6
        elif self.strategy == 'hybrid':
            # 混合: 部分基于违例，部分随机
            violation_probs = total_scores + 1e-6
            random_probs = torch.ones(n_nodes, device=device)
            probs = self.violation_weight * violation_probs + self.random_weight * random_probs
        else:
            probs = torch.ones(n_nodes, device=device)
        
        # 归一化
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = torch.ones(n_nodes, device=device)
            probs_sum = probs.sum()
        probs = probs / probs_sum
        return probs
    
    def _compute_total_score(self, cic_scores: torch.Tensor) -> torch.Tensor:
        """
        计算CIC总分 (风险放大聚合)
        
        S(e) = 1 - Π_k (1 - w_k * v_k(e))
        """
        # 默认权重
        weights = torch.tensor([0.25, 0.25, 0.25, 0.25], device=cic_scores.device)
        
        # 逐元素计算 1 - w_k * v_k
        weighted = 1.0 - weights.unsqueeze(0) * cic_scores.clamp(0.0, 1.0)  # [n_nodes, 4]
        
        # 乘积
        prod = weighted.prod(dim=1)  # [n_nodes]
        
        # 返回 1 - prod
        return 1.0 - prod
    
    def sample_mask_nodes(self, 
                          g: dgl.DGLGraph,
                          cic_scores: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样需要掩码的节点
        
        Args:
            g: DGL图
            cic_scores: CIC分数
            
        Returns:
            mask_nodes: 被掩码的节点索引
            keep_nodes: 保留的节点索引
        """
        n_nodes = g.num_nodes()
        num_mask = int(self.mask_rate * n_nodes)
        
        if num_mask == 0:
            # 不掩码任何节点
            device = g.device
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.arange(n_nodes, device=device),
            )
        
        device = g.device
        if self.strategy == 'random' or cic_scores is None:
            mask_nodes = torch.randperm(n_nodes, device=device)[:num_mask]
        else:
            probs = self.compute_mask_probs(g, cic_scores)
            mask_nodes = torch.multinomial(probs, num_mask, replacement=False)

        mask_indicator = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        mask_indicator[mask_nodes] = True
        keep_nodes = (~mask_indicator).nonzero(as_tuple=False).squeeze(1)
        
        return mask_nodes, keep_nodes
    
    def forward(self, 
                g: dgl.DGLGraph, 
                features: torch.Tensor,
                cic_scores: Optional[torch.Tensor] = None) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
        """
        应用掩码
        
        Args:
            g: DGL图
            features: 节点特征 [n_nodes, feature_dim]
            cic_scores: CIC分数 [n_nodes, 4] (可选)
            
        Returns:
            masked_g: 掩码后的图 (克隆)
            mask_nodes: 被掩码的节点索引
            keep_nodes: 保留的节点索引
        """
        if features.dim() != 2:
            raise ValueError(f"features must be 2D [n_nodes, d], got {tuple(features.shape)}")
        if features.size(0) != g.num_nodes():
            raise ValueError(f"features n_nodes mismatch: {features.size(0)} vs {g.num_nodes()}")

        # 确保mask_token维度正确
        if self.mask_token.size(1) != features.size(1):
            self.init_mask_token(features.size(1))
        
        # 采样掩码节点
        mask_nodes, keep_nodes = self.sample_mask_nodes(g, cic_scores)
        
        # 克隆图
        new_g = g.clone()
        
        # 应用掩码
        if len(mask_nodes) > 0:
            new_features = features.clone()
            new_features[mask_nodes] = self.mask_token.expand(len(mask_nodes), -1)
        else:
            new_features = features
        new_g.ndata['attr'] = new_features
        
        return new_g, mask_nodes, keep_nodes


class AttributeTypeMasking(nn.Module):
    """
    基于属性类型的差异化掩码
    
    不同类型的属性使用不同的掩码和重构策略：
    - 身份相关 (离散): uid, gid, mnt_ns, pid_ns
    - 路径相关 (序列): exe_path, cmdline, filename  
    - 时序相关 (连续): timestamps
    - 边类型 (离散): edge_type
    """
    
    IDENTITY_FIELDS = ['uid', 'gid', 'mnt_ns', 'pid_ns', 'user_ns']
    PATH_FIELDS = ['exe_path', 'cmdline', 'filename']
    TEMPORAL_FIELDS = ['timestamp', 'start_timestamp', 'delta_t']
    EDGE_FIELDS = ['edge_type']
    
    def __init__(self, 
                 mask_identity_rate: float = 0.3,
                 mask_path_rate: float = 0.2,
                 mask_temporal_rate: float = 0.4,
                 mask_edge_rate: float = 0.2):
        """
        初始化属性类型掩码
        
        Args:
            mask_identity_rate: 身份属性掩码率
            mask_path_rate: 路径属性掩码率
            mask_temporal_rate: 时序属性掩码率
            mask_edge_rate: 边类型掩码率
        """
        super().__init__()
        self.mask_identity_rate = mask_identity_rate
        self.mask_path_rate = mask_path_rate
        self.mask_temporal_rate = mask_temporal_rate
        self.mask_edge_rate = mask_edge_rate
    
    def mask_node_attributes(self, 
                              g: dgl.DGLGraph,
                              attr_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        掩码节点属性
        
        Args:
            g: DGL图
            attr_dict: 属性字典 {attr_name: tensor}
            
        Returns:
            masked_attrs: 掩码后的属性字典
            mask_indicators: 掩码指示器 {attr_name: bool_tensor}
        """
        masked_attrs = {}
        mask_indicators = {}
        n_nodes = g.num_nodes()
        
        for attr_name, attr_tensor in attr_dict.items():
            # 确定掩码率
            if attr_name in self.IDENTITY_FIELDS:
                rate = self.mask_identity_rate
            elif attr_name in self.PATH_FIELDS:
                rate = self.mask_path_rate
            elif attr_name in self.TEMPORAL_FIELDS:
                rate = self.mask_temporal_rate
            else:
                rate = 0.1  # 默认较低的掩码率
            
            # 生成掩码
            mask = torch.rand(n_nodes, device=attr_tensor.device) < rate
            mask_indicators[attr_name] = mask
            
            # 应用掩码 (用0或特殊值替换)
            masked = attr_tensor.clone()
            if masked.dtype in [torch.float32, torch.float64]:
                masked[mask] = 0.0
            else:
                masked[mask] = 0
            masked_attrs[attr_name] = masked
        
        return masked_attrs, mask_indicators


class CausalChainMasking(nn.Module):
    """
    基于因果链的掩码策略
    
    沿着因果链掩码关键节点，迫使模型学习因果推理能力。
    """
    
    def __init__(self, 
                 mask_rate: float = 0.3,
                 chain_length: int = 3):
        """
        初始化因果链掩码
        
        Args:
            mask_rate: 链上节点掩码率
            chain_length: 最大链长度
        """
        super().__init__()
        self.mask_rate = mask_rate
        self.chain_length = chain_length
    
    def find_causal_chains(self, g: dgl.DGLGraph, seed_nodes: torch.Tensor) -> List[List[int]]:
        """
        从种子节点出发沿边找因果链
        
        Args:
            g: DGL图
            seed_nodes: 种子节点索引
            
        Returns:
            chains: 因果链列表
        """
        chains = []
        
        for seed in seed_nodes.tolist():
            chain = [seed]
            current = seed
            
            for _ in range(self.chain_length - 1):
                # 获取出边的目标节点
                successors = g.successors(current).tolist()
                if not successors:
                    break
                # 随机选择一个后继
                next_node = random.choice(successors)
                chain.append(next_node)
                current = next_node
            
            if len(chain) > 1:
                chains.append(chain)
        
        return chains
    
    def forward(self, 
                g: dgl.DGLGraph,
                features: torch.Tensor,
                cic_scores: Optional[torch.Tensor] = None) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        """
        应用因果链掩码
        
        Args:
            g: DGL图
            features: 节点特征
            cic_scores: CIC分数 (用于选择种子)
            
        Returns:
            masked_g: 掩码后的图
            masked_nodes: 所有被掩码的节点
        """
        n_nodes = g.num_nodes()
        if n_nodes == 0:
            return g.clone(), torch.empty(0, dtype=torch.long, device=features.device)
        
        # 选择种子节点 (优先选择高违例分数的节点)
        if cic_scores is not None:
            if cic_scores.dim() == 2:
                total_scores = 1.0 - (1.0 - 0.25 * cic_scores).prod(dim=1)
            else:
                total_scores = cic_scores
            
            # 按分数排序，取top-k作为种子
            num_seeds = max(1, int(n_nodes * self.mask_rate / self.chain_length))
            _, seed_indices = torch.topk(total_scores, min(num_seeds, n_nodes))
            seed_nodes = seed_indices.to(device=g.device)
        else:
            # 随机选择种子
            num_seeds = max(1, int(n_nodes * self.mask_rate / self.chain_length))
            seed_nodes = torch.randperm(n_nodes, device=g.device)[:num_seeds]
        
        # 找因果链
        chains = self.find_causal_chains(g, seed_nodes)
        
        # 收集所有需要掩码的节点
        all_mask_nodes = set()
        for chain in chains:
            # 掩码链上除最后一个节点外的所有节点 (保留目标用于重构)
            for node in chain[:-1]:
                all_mask_nodes.add(node)
        
        mask_nodes = torch.tensor(list(all_mask_nodes), dtype=torch.long, device=g.device)
        
        # 克隆并掩码
        new_g = g.clone()
        if len(mask_nodes) > 0:
            new_features = features.clone()
            new_features[mask_nodes] = 0  # 简单置零掩码
            new_g.ndata['attr'] = new_features
        else:
            new_g.ndata['attr'] = features
        
        return new_g, mask_nodes


# ============================================================================
# 工厂函数
# ============================================================================

def create_masking_module(strategy: str = 'hybrid', 
                          mask_rate: float = 0.5,
                          **kwargs) -> nn.Module:
    """
    创建掩码模块
    
    Args:
        strategy: 掩码策略
        mask_rate: 掩码率
        **kwargs: 其他参数
        
    Returns:
        掩码模块实例
    """
    if strategy in ['random', 'violation', 'inverse_violation', 'hybrid']:
        return InvariantAwareMasking(
            mask_rate=mask_rate,
            strategy=strategy,
            **kwargs
        )
    elif strategy == 'causal_chain':
        return CausalChainMasking(
            mask_rate=mask_rate,
            **kwargs
        )
    elif strategy == 'attribute':
        return AttributeTypeMasking(**kwargs)
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")
