"""
单调融合与排序一致性模块 (Monotonic Fusion & Ranking Consistency)

研究路线 Phase 3:
- MonotonicFusion: 使用softplus确保权重正数，保证融合的单调性
- RankingConsistencyLoss: pairwise ranking loss确保异常节点排序高于正常节点
- 风险放大聚合: S(e) = 1 - Π_k(1 - w_k * v_k(e))

与以下模块对接:
- utils/cic_invariants.py: InvariantScores (i_reach, i_creator, i_timing, i_alias)
- model/contrastive.py: 对比学习分数
- model/masking.py: 掩码重构误差

参考: implementation_plan.md Phase 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


def _normalize_unit_interval(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).to(dtype=torch.float32)
    if x.numel() == 0:
        return x
    x_min = x.min()
    x_max = x.max()
    denom = x_max - x_min
    if float(denom) > 1e-12:
        x = (x - x_min) / (denom + 1e-12)
    else:
        x = torch.zeros_like(x)
    return x.clamp(0.0, 1.0)


class MonotonicFusion(nn.Module):
    """
    单调融合模块
    
    确保各不变量分数的融合保持单调性：任意单项分数增加都会导致总分增加。
    使用可学习的正权重（通过softplus保证）。
    
    与CIC模块对接:
    - 输入: invariant_scores [batch, 4] 对应 [I_reach, I_creator, I_timing, I_alias]
    - 输出: fused_score [batch]
    """
    
    # 不变量名称顺序（与InvariantScores.to_vector()一致）
    INVARIANT_NAMES = ['i_reach', 'i_creator', 'i_timing', 'i_alias']
    
    def __init__(self, 
                 n_invariants: int = 4,
                 init_weights: Optional[List[float]] = None,
                 fusion_type: str = 'risk_amplification'):
        """
        初始化单调融合模块
        
        Args:
            n_invariants: 不变量数量 (默认4)
            init_weights: 初始权重 (默认均匀)
            fusion_type: 融合类型
                - 'weighted_sum': 加权和
                - 'risk_amplification': 风险放大聚合 S = 1 - Π(1 - w_k * v_k)
                - 'max': 取最大值
        """
        super().__init__()
        self.n_invariants = n_invariants
        self.fusion_type = fusion_type
        
        if init_weights is None:
            init_weights = [1.0] * n_invariants
        
        # 使用raw参数，通过softplus转换确保权重为正
        self.raw_weights = nn.Parameter(torch.tensor(init_weights))
        
        # 可选的偏置项（用于调整基线）
        self.bias = nn.Parameter(torch.zeros(1))
        
        # 归一化权重的温度参数
        self.temperature = nn.Parameter(torch.ones(1))
    
    @property
    def weights(self) -> torch.Tensor:
        """获取正的权重（通过softplus）"""
        return F.softplus(self.raw_weights)
    
    @property
    def normalized_weights(self) -> torch.Tensor:
        """获取归一化的权重（和为1）"""
        w = self.weights
        return w / (w.sum() + 1e-8)
    
    def forward(self,
                invariant_scores: torch.Tensor,
                normalize: bool = False) -> torch.Tensor:
        """
        融合不变量分数
        
        Args:
            invariant_scores: 不变量分数 [batch, n_invariants] 或 [n_nodes, n_invariants]
                              顺序: [i_reach, i_creator, i_timing, i_alias]
            normalize: 是否使用归一化权重
            
        Returns:
            fused_score: 融合后的分数 [batch] 或 [n_nodes]
        """
        if invariant_scores.dim() == 1:
            invariant_scores = invariant_scores.unsqueeze(0)
            squeeze_out = True
        else:
            squeeze_out = False

        if invariant_scores.size(-1) != self.n_invariants:
            raise ValueError(
                f"invariant_scores last dim mismatch: got {invariant_scores.size(-1)}, expected {self.n_invariants}"
            )

        invariant_scores = torch.nan_to_num(invariant_scores, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        
        # 获取权重
        if normalize:
            weights = self.normalized_weights
        else:
            weights = self.weights
        
        # 确保权重在合理范围内（对于risk_amplification，权重应在[0,1]）
        if self.fusion_type == 'risk_amplification':
            weights = weights.clamp(0.0, 1.0)
        
        weights_view = weights.view(*([1] * (invariant_scores.dim() - 1)), -1)

        if self.fusion_type == 'weighted_sum':
            # 加权和: S = Σ w_k * v_k
            fused = (invariant_scores * weights_view).sum(dim=-1)

        elif self.fusion_type == 'risk_amplification':
            # 风险放大聚合: S = 1 - Π_k (1 - w_k * v_k)
            # 与 InvariantScores.total_score() 保持一致
            weighted = 1.0 - weights_view * invariant_scores  # [..., n_inv]
            prod = weighted.prod(dim=-1)  # [batch]
            fused = 1.0 - prod

        elif self.fusion_type == 'max':
            # 取最大值: S = max_k(w_k * v_k)
            weighted_scores = invariant_scores * weights_view
            fused = weighted_scores.max(dim=-1).values
            
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        # 添加偏置并限制范围
        fused = (fused + self.bias).clamp(0.0, 1.0)
        
        return fused.squeeze(0) if squeeze_out else fused
    
    def get_weight_dict(self) -> Dict[str, float]:
        """获取每个不变量的权重字典"""
        weights = self.normalized_weights.detach().cpu().tolist()
        return {name: w for name, w in zip(self.INVARIANT_NAMES, weights)}


class RankingConsistencyLoss(nn.Module):
    """
    排序一致性损失
    
    确保异常节点（高CIC分数）的预测分数高于正常节点。
    使用pairwise margin ranking loss。
    
    与CIC模块对接:
    - pred_scores: 模型预测的异常分数
    - cic_scores: CIC不变量分数（作为软标签或阈值划分）
    """
    
    def __init__(self, 
                 margin: float = 1.0,
                 threshold: float = 0.5,
                 use_soft_labels: bool = True,
                 max_pairs: int = 10000):
        """
        初始化排序一致性损失
        
        Args:
            margin: ranking loss的边界值
            threshold: 区分正常/异常的CIC分数阈值
            use_soft_labels: 是否使用软标签（连续分数）而非硬划分
            max_pairs: 最大采样的节点对数（避免O(N^2)内存爆炸）
        """
        super().__init__()
        self.margin = margin
        self.threshold = threshold
        self.use_soft_labels = use_soft_labels
        self.max_pairs = max_pairs
    
    def forward(self,
                pred_scores: torch.Tensor,
                cic_scores: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算排序一致性损失
        
        Args:
            pred_scores: 模型预测的异常分数 [n_nodes]
            cic_scores: CIC分数 [n_nodes] 或 [n_nodes, 4]
            labels: 可选的真实标签 [n_nodes] (0=正常, 1=异常)
            
        Returns:
            loss: 排序一致性损失
        """
        pred_scores = pred_scores.reshape(-1)
        device = pred_scores.device
        n_nodes = pred_scores.numel()
        
        if n_nodes < 2:
            return torch.tensor(0.0, device=device)
        
        # 处理CIC分数
        if cic_scores.dim() == 2:
            # 使用风险放大聚合计算总分
            weights = torch.tensor([0.25, 0.25, 0.25, 0.25], device=cic_scores.device)
            cic_total = 1.0 - (1.0 - weights.unsqueeze(0) * cic_scores.clamp(0, 1)).prod(dim=-1)
        else:
            cic_total = cic_scores.clamp(0, 1)

        # 处理NaN
        cic_total = torch.nan_to_num(cic_total, nan=0.0, posinf=1.0, neginf=0.0).reshape(-1).to(device=device)

        if cic_total.numel() != n_nodes:
            raise ValueError(f"cic_scores length mismatch: got {cic_total.numel()}, expected {n_nodes}")
        
        if self.use_soft_labels and labels is None:
            # 软标签模式: 直接使用CIC分数作为排序依据
            return self._soft_ranking_loss(pred_scores, cic_total)
        else:
            # 硬标签模式: 使用阈值或真实标签划分
            if labels is None:
                labels = (cic_total > self.threshold).long()
            return self._hard_ranking_loss(pred_scores, labels)
    
    def _soft_ranking_loss(self, 
                           pred_scores: torch.Tensor, 
                           cic_scores: torch.Tensor) -> torch.Tensor:
        """
        软标签排序损失
        
        希望: CIC分数高的节点，预测分数也应该高
        使用采样避免O(N^2)
        """
        device = pred_scores.device
        n = pred_scores.size(0)
        
        # 随机采样节点对
        n_pairs = min(self.max_pairs, n * (n - 1) // 2)
        
        if n_pairs == 0:
            return torch.tensor(0.0, device=device)
        
        # 采样
        idx_i = torch.randint(0, n, (n_pairs,), device=device)
        idx_j = torch.randint(0, n, (n_pairs,), device=device)
        
        # 确保i != j
        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]
        
        if len(idx_i) == 0:
            return torch.tensor(0.0, device=device)
        
        # 计算CIC分数差和预测分数差
        cic_diff = cic_scores[idx_i] - cic_scores[idx_j]  # 希望符号一致
        pred_diff = pred_scores[idx_i] - pred_scores[idx_j]
        
        # 损失: 如果cic_diff > 0，则pred_diff也应该 > 0
        # margin ranking loss: max(0, margin - sign * (pred_i - pred_j))
        sign = torch.sign(cic_diff)
        
        # 忽略CIC分数相差太小的对
        significant_mask = cic_diff.abs() > 0.1
        if significant_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        sign = sign[significant_mask]
        pred_diff = pred_diff[significant_mask]
        
        loss = F.relu(self.margin - sign * pred_diff)
        return loss.mean()
    
    def _hard_ranking_loss(self,
                           pred_scores: torch.Tensor,
                           labels: torch.Tensor) -> torch.Tensor:
        """
        硬标签排序损失
        
        希望: 异常节点(label=1)的预测分数高于正常节点(label=0)
        """
        device = pred_scores.device
        
        labels = labels.reshape(-1).to(device=pred_scores.device)
        if labels.numel() != pred_scores.numel():
            raise ValueError(f"labels length mismatch: got {labels.numel()}, expected {pred_scores.numel()}")

        pos_mask = labels == 1  # 异常
        neg_mask = labels == 0  # 正常
        
        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()
        
        if n_pos == 0 or n_neg == 0:
            return torch.tensor(0.0, device=device)
        
        pos_scores = pred_scores[pos_mask]  # [n_pos]
        neg_scores = pred_scores[neg_mask]  # [n_neg]
        
        # 采样避免O(n_pos * n_neg)
        n_pairs = min(self.max_pairs, n_pos * n_neg)
        
        if n_pairs < n_pos * n_neg:
            # 需要采样
            pos_idx = torch.randint(0, n_pos, (n_pairs,), device=device)
            neg_idx = torch.randint(0, n_neg, (n_pairs,), device=device)
            
            sampled_pos = pos_scores[pos_idx]
            sampled_neg = neg_scores[neg_idx]
            
            # margin loss: max(0, margin - (pos - neg))
            loss = F.relu(self.margin - (sampled_pos - sampled_neg))
        else:
            # 可以计算所有对
            # [n_pos, 1] - [1, n_neg] = [n_pos, n_neg]
            diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            loss = F.relu(self.margin - diff)
        
        return loss.mean()


class MultiSourceFusion(nn.Module):
    """
    多源分数融合模块
    
    融合来自不同模块的异常分数:
    - CIC不变量分数 (from cic_invariants.py)
    - 对比学习分数 (from contrastive.py)
    - 重构误差分数 (from autoencoder)
    """
    
    def __init__(self, 
                 n_sources: int = 3,
                 fusion_type: str = 'learned'):
        """
        初始化多源融合
        
        Args:
            n_sources: 分数来源数量
            fusion_type: 融合方式
                - 'mean': 简单平均
                - 'max': 取最大
                - 'learned': 可学习加权
        """
        super().__init__()
        self.n_sources = n_sources
        self.fusion_type = fusion_type
        
        if fusion_type == 'learned':
            self.raw_weights = nn.Parameter(torch.ones(n_sources))
    
    @property
    def weights(self) -> torch.Tensor:
        if self.fusion_type == 'learned':
            w = F.softplus(self.raw_weights)
            return w / (w.sum() + 1e-8)
        else:
            return torch.ones(self.n_sources) / self.n_sources
    
    def forward(self,
                cic_score: torch.Tensor,
                contrastive_score: Optional[torch.Tensor] = None,
                recon_error: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        融合多源分数
        
        Args:
            cic_score: CIC不变量分数 [n_nodes]
            contrastive_score: 对比学习异常分数 [n_nodes] (可选)
            recon_error: 重构误差 [n_nodes] (可选)
            
        Returns:
            fused_score: 融合后的异常分数 [n_nodes]
        """
        if cic_score.dim() != 1:
            raise ValueError(f"cic_score must be 1D [n_nodes], got shape {tuple(cic_score.shape)}")
        cic_score = torch.nan_to_num(cic_score, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        device = cic_score.device

        # 收集所有可用的分数
        scores = [cic_score]
        if contrastive_score is not None:
            cs = contrastive_score.to(device)
            if cs.numel() == 1:
                cs = cs.reshape(1).expand_as(cic_score)
            elif cs.dim() != 1 or cs.numel() != cic_score.numel():
                raise ValueError(
                    f"contrastive_score shape mismatch: got {tuple(cs.shape)}, expected {(cic_score.numel(),)}"
                )
            scores.append(_normalize_unit_interval(cs))
        if recon_error is not None:
            err = recon_error.to(device)
            if err.numel() == 1:
                err = err.reshape(1).expand_as(cic_score)
            elif err.dim() != 1 or err.numel() != cic_score.numel():
                raise ValueError(f"recon_error shape mismatch: got {tuple(err.shape)}, expected {(cic_score.numel(),)}")
            scores.append(_normalize_unit_interval(err))
        
        # 补齐到n_sources
        while len(scores) < self.n_sources:
            scores.append(torch.zeros_like(cic_score))
        
        # Stack并融合
        stacked = torch.stack(scores[:self.n_sources], dim=-1)  # [n_nodes, n_sources]
        
        if self.fusion_type == 'mean':
            fused = stacked.mean(dim=-1)
        elif self.fusion_type == 'max':
            fused = stacked.max(dim=-1).values
        elif self.fusion_type == 'learned':
            weights = self.weights.to(device)
            fused = (stacked * weights.unsqueeze(0)).sum(dim=-1)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        return fused.clamp(0.0, 1.0)


class AnomalyScorer(nn.Module):
    """
    异常评分模块
    
    结合单调融合、排序一致性损失、多源融合的完整异常评分Pipeline。
    """
    
    def __init__(self,
                 n_invariants: int = 4,
                 n_sources: int = 3,
                 margin: float = 1.0,
                 invariant_fusion_type: str = 'risk_amplification',
                 source_fusion_type: str = 'learned'):
        """
        初始化异常评分器
        
        Args:
            n_invariants: 不变量数量
            n_sources: 分数来源数量
            margin: 排序损失边界
            invariant_fusion_type: 不变量融合方式
            source_fusion_type: 多源融合方式
        """
        super().__init__()
        
        # 单调融合（不变量级别）
        self.invariant_fusion = MonotonicFusion(
            n_invariants=n_invariants,
            fusion_type=invariant_fusion_type
        )
        
        # 多源融合
        self.source_fusion = MultiSourceFusion(
            n_sources=n_sources,
            fusion_type=source_fusion_type
        )
        
        # 排序一致性损失
        self.ranking_loss = RankingConsistencyLoss(margin=margin)
    
    def compute_anomaly_score(self,
                               invariant_scores: torch.Tensor,
                               contrastive_score: Optional[torch.Tensor] = None,
                               recon_error: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算异常分数
        
        Args:
            invariant_scores: 不变量分数 [n_nodes, 4]
            contrastive_score: 对比学习分数 [n_nodes]
            recon_error: 重构误差 [n_nodes]
            
        Returns:
            anomaly_score: 最终异常分数 [n_nodes]
        """
        # 1. 融合CIC不变量
        cic_fused = self.invariant_fusion(invariant_scores)
        
        # 2. 多源融合
        final_score = self.source_fusion(cic_fused, contrastive_score, recon_error)
        
        return final_score
    
    def compute_ranking_loss(self,
                              pred_scores: torch.Tensor,
                              cic_scores: torch.Tensor,
                              labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算排序一致性损失
        """
        return self.ranking_loss(pred_scores, cic_scores, labels)
    
    def forward(self,
                invariant_scores: torch.Tensor,
                contrastive_score: Optional[torch.Tensor] = None,
                recon_error: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播（训练模式）
        
        Returns:
            包含anomaly_score和ranking_loss的字典
        """
        # 计算异常分数
        anomaly_score = self.compute_anomaly_score(
            invariant_scores, contrastive_score, recon_error
        )
        
        # 计算排序损失
        ranking_loss = self.compute_ranking_loss(
            anomaly_score, invariant_scores, labels
        )
        
        return {
            'anomaly_score': anomaly_score,
            'ranking_loss': ranking_loss,
            'cic_fused': self.invariant_fusion(invariant_scores),
            'weights': self.invariant_fusion.get_weight_dict()
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_anomaly_scorer(**kwargs) -> AnomalyScorer:
    """创建异常评分器"""
    return AnomalyScorer(**kwargs)


def create_ranking_loss(margin: float = 1.0, 
                        threshold: float = 0.5,
                        use_soft_labels: bool = True) -> RankingConsistencyLoss:
    """创建排序一致性损失"""
    return RankingConsistencyLoss(
        margin=margin,
        threshold=threshold,
        use_soft_labels=use_soft_labels
    )
