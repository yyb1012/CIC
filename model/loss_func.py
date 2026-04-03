"""
损失函数模块

包含:
- sce_loss: Scaled Cosine Error Loss (原有)
- ranking_consistency_loss: 排序一致性损失 (Phase 3)
- combined_cic_loss: 组合损失 (重构 + 对比 + 排序)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def sce_loss(x, y, alpha=3):
    """
    Scaled Cosine Error Loss
    
    Args:
        x: 预测特征
        y: 目标特征
        alpha: 缩放指数
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def ranking_consistency_loss(pred_scores: torch.Tensor,
                              cic_scores: torch.Tensor,
                              margin: float = 1.0,
                              max_pairs: int = 10000) -> torch.Tensor:
    """
    排序一致性损失 (快速版本)
    
    确保: CIC分数高的节点，预测分数也应该高
    
    Args:
        pred_scores: 模型预测分数 [n_nodes]
        cic_scores: CIC分数 [n_nodes] 或 [n_nodes, 4]
        margin: 排序边界
        max_pairs: 最大采样对数
        
    Returns:
        loss: 排序损失
    """
    pred_scores = pred_scores.reshape(-1)
    device = pred_scores.device
    n = pred_scores.numel()
    
    if n < 2:
        return torch.tensor(0.0, device=device)
    
    # 处理4维CIC分数
    if cic_scores.dim() == 2:
        weights = torch.tensor([0.25, 0.25, 0.25, 0.25], device=cic_scores.device)
        cic_total = 1.0 - (1.0 - weights.unsqueeze(0) * cic_scores.clamp(0, 1)).prod(dim=-1)
    else:
        cic_total = cic_scores.clamp(0, 1)

    cic_total = torch.nan_to_num(cic_total, nan=0.0, posinf=1.0, neginf=0.0).reshape(-1).to(device=device)
    if cic_total.numel() != n:
        raise ValueError(f"cic_scores length mismatch: got {cic_total.numel()}, expected {n}")
    
    # 采样节点对
    n_pairs = min(int(max_pairs), n * (n - 1) // 2)
    
    if n_pairs == 0:
        return torch.tensor(0.0, device=device)
    
    idx_i = torch.randint(0, n, (n_pairs,), device=device)
    idx_j = torch.randint(0, n, (n_pairs,), device=device)
    
    # 去除相同节点对
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    
    if len(idx_i) == 0:
        return torch.tensor(0.0, device=device)
    
    # 计算分数差
    cic_diff = cic_total[idx_i] - cic_total[idx_j]
    pred_diff = pred_scores[idx_i] - pred_scores[idx_j]
    
    # 只关注显著差异的对
    significant = cic_diff.abs() > 0.1
    if significant.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    sign = torch.sign(cic_diff[significant])
    pred_diff_sig = pred_diff[significant]
    
    # Margin ranking loss
    loss = F.relu(margin - sign * pred_diff_sig)
    return loss.mean()


def combined_cic_loss(recon_loss: torch.Tensor,
                       contrastive_loss: Optional[torch.Tensor] = None,
                       ranking_loss: Optional[torch.Tensor] = None,
                       recon_weight: float = 1.0,
                       contrastive_weight: float = 0.5,
                       ranking_weight: float = 0.3) -> Dict[str, torch.Tensor]:
    """
    组合CIC研究路线的所有损失
    
    Args:
        recon_loss: 重构损失 (掩码自监督)
        contrastive_loss: 对比学习损失 (可选)
        ranking_loss: 排序一致性损失 (可选)
        recon_weight: 重构损失权重
        contrastive_weight: 对比损失权重
        ranking_weight: 排序损失权重
        
    Returns:
        损失字典，包含各项损失和总损失
    """
    device = recon_loss.device
    losses = {'recon': recon_loss * recon_weight}
    
    total = losses['recon']
    
    if contrastive_loss is not None:
        losses['contrastive'] = contrastive_loss * contrastive_weight
        total = total + losses['contrastive']
    
    if ranking_loss is not None:
        losses['ranking'] = ranking_loss * ranking_weight
        total = total + losses['ranking']
    
    losses['total'] = total
    
    return losses


class CombinedLoss(nn.Module):
    """
    组合损失模块
    
    整合CIC研究路线的所有损失项，支持可学习的权重。
    """
    
    def __init__(self,
                 recon_weight: float = 1.0,
                 contrastive_weight: float = 0.5,
                 ranking_weight: float = 0.3,
                 learnable_weights: bool = False):
        """
        Args:
            recon_weight: 重构损失权重
            contrastive_weight: 对比损失权重
            ranking_weight: 排序损失权重
            learnable_weights: 是否使用可学习权重
        """
        super().__init__()
        self.learnable_weights = learnable_weights
        
        if learnable_weights:
            self.raw_weights = nn.Parameter(
                torch.tensor([recon_weight, contrastive_weight, ranking_weight])
            )
        else:
            self.register_buffer('weights', 
                torch.tensor([recon_weight, contrastive_weight, ranking_weight]))
    
    @property
    def loss_weights(self) -> torch.Tensor:
        if self.learnable_weights:
            return F.softplus(self.raw_weights)
        return self.weights
    
    def forward(self,
                recon_loss: torch.Tensor,
                contrastive_loss: Optional[torch.Tensor] = None,
                ranking_loss: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算组合损失"""
        weights = self.loss_weights
        device = recon_loss.device
        
        losses = {
            'recon': recon_loss * weights[0],
        }
        total = losses['recon']
        
        if contrastive_loss is not None:
            losses['contrastive'] = contrastive_loss * weights[1]
            total = total + losses['contrastive']
        
        if ranking_loss is not None:
            losses['ranking'] = ranking_loss * weights[2]
            total = total + losses['ranking']
        
        losses['total'] = total
        losses['weights'] = {
            'recon': weights[0].item(),
            'contrastive': weights[1].item() if contrastive_loss is not None else 0,
            'ranking': weights[2].item() if ranking_loss is not None else 0,
        }
        
        return losses
