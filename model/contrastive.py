"""
对比学习模块 (Contrastive Learning)

基于CIC不变量的对比学习：
- 正样本: 保持不变量的良性子图
- 负样本: 违反不变量的扰动子图
- 损失函数: InfoNCE, Triplet Loss等

参考: 研究路线中的对比学习部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import dgl

from .negative_sampling import NegativeSampleBuilder, FeaturePerturbation


class CICContrastiveLearning(nn.Module):
    """
    基于CIC不变量的对比学习模块
    
    通过正负样本对比，使模型学会：
    1. 将正常模式的表示聚集在一起
    2. 将异常模式的表示与正常模式分离
    """
    
    def __init__(self,
                 hidden_dim: int,
                 projection_dim: int = 128,
                 temperature: float = 0.07,
                 negative_builder: Optional[NegativeSampleBuilder] = None):
        """
        初始化对比学习模块
        
        Args:
            hidden_dim: 编码器输出的隐藏维度
            projection_dim: 投影空间维度
            temperature: InfoNCE温度参数
            negative_builder: 负样本构建器
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # 投影头: 将编码器输出映射到对比空间
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # 负样本构建器
        if negative_builder is None:
            negative_builder = NegativeSampleBuilder()
        self.negative_builder = negative_builder
        
        # 特征扰动器 (用于在线生成负样本)
        self.feature_perturber = FeaturePerturbation()
    
    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        将嵌入投影到对比空间
        
        Args:
            embeddings: 编码器输出 [batch, hidden_dim] 或 [n_nodes, hidden_dim]
            
        Returns:
            projected: 投影后的嵌入 [batch, projection_dim]
        """
        return F.normalize(self.projector(embeddings), dim=-1)
    
    def infonce_loss(self,
                     anchor: torch.Tensor,
                     positive: torch.Tensor,
                     negatives: torch.Tensor) -> torch.Tensor:
        """
        计算InfoNCE损失
        
        Args:
            anchor: 锚点嵌入 [batch, dim]
            positive: 正样本嵌入 [batch, dim]
            negatives: 负样本嵌入 [batch, n_neg, dim] 或 [n_neg, dim]
            
        Returns:
            loss: InfoNCE损失
        """
        batch_size = anchor.size(0)
        
        # 确保negatives是3D
        if negatives.dim() == 2:
            negatives = negatives.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 正样本相似度
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # [batch]
        
        # 负样本相似度
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature  # [batch, n_neg]
        
        # 分母: exp(pos) + sum(exp(neg))
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [batch, 1+n_neg]
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)  # 正样本在第0位
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def triplet_loss(self,
                     anchor: torch.Tensor,
                     positive: torch.Tensor,
                     negative: torch.Tensor,
                     margin: float = 1.0) -> torch.Tensor:
        """
        计算Triplet损失
        
        Args:
            anchor: 锚点嵌入
            positive: 正样本嵌入
            negative: 负样本嵌入
            margin: 边界值
            
        Returns:
            loss: Triplet损失
        """
        pos_dist = torch.norm(anchor - positive, dim=-1)
        neg_dist = torch.norm(anchor - negative, dim=-1)
        
        loss = F.relu(pos_dist - neg_dist + margin)
        return loss.mean()
    
    def forward(self,
                encoder: nn.Module,
                g: dgl.DGLGraph,
                features: torch.Tensor,
                cic_scores: Optional[torch.Tensor] = None,
                loss_type: str = 'infonce') -> torch.Tensor:
        """
        计算对比学习损失
        
        Args:
            encoder: 图编码器
            g: 输入图
            features: 节点特征
            cic_scores: CIC分数
            loss_type: 损失类型 ('infonce' 或 'triplet')
            
        Returns:
            loss: 对比损失
        """
        # 获取锚点嵌入 (原始图)
        anchor_embed = encoder(g, features)
        if isinstance(anchor_embed, tuple):
            anchor_embed = anchor_embed[0]  # 如果返回多个值，取第一个
        
        # 图级表示 (mean pooling)
        anchor_repr = anchor_embed.mean(dim=0, keepdim=True)  # [1, hidden]
        anchor_proj = self.project(anchor_repr)  # [1, proj_dim]
        
        # 正样本: 原图的另一个视图 (可以是掩码后重构的)
        # 这里简单使用添加小噪声的版本
        pos_features = features + torch.randn_like(features) * 0.01
        pos_embed = encoder(g, pos_features)
        if isinstance(pos_embed, tuple):
            pos_embed = pos_embed[0]
        pos_repr = pos_embed.mean(dim=0, keepdim=True)
        pos_proj = self.project(pos_repr)
        
        # 负样本: 使用特征扰动生成
        neg_features = self.feature_perturber(features, cic_scores)
        neg_embed = encoder(g, neg_features)
        if isinstance(neg_embed, tuple):
            neg_embed = neg_embed[0]
        neg_repr = neg_embed.mean(dim=0, keepdim=True)
        neg_proj = self.project(neg_repr)
        
        # 计算损失
        if loss_type == 'infonce':
            # 需要多个负样本
            negatives = neg_proj.unsqueeze(0)  # [1, 1, proj_dim]
            loss = self.infonce_loss(anchor_proj, pos_proj, negatives)
        else:
            loss = self.triplet_loss(anchor_proj, pos_proj, neg_proj)
        
        return loss


class NodeLevelContrastive(nn.Module):
    """
    节点级对比学习
    
    对比同一图中的正常节点和异常节点。
    """
    
    def __init__(self,
                 hidden_dim: int,
                 temperature: float = 0.07):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self,
                node_embeddings: torch.Tensor,
                cic_scores: torch.Tensor,
                threshold: float = 0.5) -> torch.Tensor:
        """
        计算节点级对比损失
        
        Args:
            node_embeddings: 节点嵌入 [n_nodes, hidden]
            cic_scores: CIC分数 [n_nodes] 或 [n_nodes, 4]
            threshold: 区分正常/异常的阈值
            
        Returns:
            loss: 对比损失
        """
        # 计算总分
        device = node_embeddings.device
        if cic_scores.dim() == 2:
            cic_scores_clean = torch.nan_to_num(cic_scores.to(device), nan=0.0, posinf=1.0, neginf=0.0)
            total_scores = 1.0 - (1.0 - 0.25 * cic_scores_clean).prod(dim=1)
        else:
            total_scores = cic_scores
        total_scores = torch.nan_to_num(total_scores.to(device), nan=0.0, posinf=1.0, neginf=0.0)
        
        # 划分正常节点和高分节点
        normal_mask = total_scores < threshold
        anomaly_mask = total_scores >= threshold
        
        if normal_mask.sum() == 0 or anomaly_mask.sum() == 0:
            return torch.tensor(0.0, device=node_embeddings.device)
        
        # 投影
        proj = self.projector(node_embeddings)
        proj = F.normalize(proj, dim=-1)
        
        normal_proj = proj[normal_mask]  # [n_normal, dim]
        anomaly_proj = proj[anomaly_mask]  # [n_anomaly, dim]

        # 避免 O(N^2) 爆显存：对过多节点做随机子采样（不影响接口正确性，训练近似）
        max_normals = 2048
        max_anomalies = 2048
        if normal_proj.size(0) > max_normals:
            idx = torch.randperm(normal_proj.size(0), device=normal_proj.device)[:max_normals]
            normal_proj = normal_proj[idx]
        if anomaly_proj.size(0) > max_anomalies:
            idx = torch.randperm(anomaly_proj.size(0), device=anomaly_proj.device)[:max_anomalies]
            anomaly_proj = anomaly_proj[idx]

        if normal_proj.size(0) < 2 or anomaly_proj.size(0) < 1:
            return torch.tensor(0.0, device=node_embeddings.device)

        # Supervised contrastive: normals pull together, push away anomalies
        sim_nn = torch.mm(normal_proj, normal_proj.t()) / self.temperature  # [Nn, Nn]
        diag = torch.eye(sim_nn.size(0), device=sim_nn.device, dtype=torch.bool)
        sim_nn = sim_nn.masked_fill(diag, float('-inf'))

        sim_na = torch.mm(normal_proj, anomaly_proj.t()) / self.temperature  # [Nn, Na]

        log_pos = torch.logsumexp(sim_nn, dim=1)  # positives: other normals
        log_den = torch.logsumexp(torch.cat([sim_nn, sim_na], dim=1), dim=1)
        loss = -(log_pos - log_den).mean()
        return loss

    @torch.no_grad()
    def anomaly_score(self,
                      node_embeddings: torch.Tensor,
                      cic_scores: torch.Tensor,
                      threshold: float = 0.5) -> torch.Tensor:
        """
        Produce a node-level anomaly score in [0, 1] for MultiSourceFusion.

        Strategy:
        - Use CIC total score to pick "normal" nodes (< threshold) as a prototype set.
        - Score each node by (1 - cosine_sim(node, normal_prototype)), then min-max normalize.
        """
        if node_embeddings.dim() != 2:
            raise ValueError(f"node_embeddings must be 2D [n_nodes, d], got {tuple(node_embeddings.shape)}")
        n_nodes = node_embeddings.size(0)
        device = node_embeddings.device

        if cic_scores.dim() == 2:
            # 先清理 NaN/Inf，防止 prod() 中出现 CUDA 错误
            cic_scores_clean = torch.nan_to_num(cic_scores.to(device), nan=0.0, posinf=1.0, neginf=0.0)
            total_scores = 1.0 - (1.0 - 0.25 * cic_scores_clean).prod(dim=1)
        else:
            total_scores = cic_scores
        total_scores = torch.nan_to_num(total_scores.to(device), nan=0.0, posinf=1.0, neginf=0.0).reshape(-1)
        if total_scores.numel() != n_nodes:
            raise ValueError(f"cic_scores length mismatch: got {total_scores.numel()}, expected {n_nodes}")

        proj = self.projector(node_embeddings)
        proj = torch.nan_to_num(proj, nan=0.0, posinf=0.0, neginf=0.0)
        proj = F.normalize(proj, dim=-1)

        normal_mask = total_scores < float(threshold)
        if normal_mask.sum() == 0:
            # fallback: take lowest-k as "normal"
            k = min(128, n_nodes)
            _, idx = torch.topk(-total_scores, k)
            normal_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
            normal_mask[idx] = True

        proto = proj[normal_mask].mean(dim=0, keepdim=True)
        proto = F.normalize(proto, dim=-1)
        sim = (proj * proto).sum(dim=-1)
        score = (1.0 - sim).clamp(0.0, 2.0)

        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        s_min = score.min()
        s_max = score.max()
        denom = s_max - s_min
        if float(denom) > 1e-12:
            score = (score - s_min) / (denom + 1e-12)
        else:
            score = torch.zeros_like(score)
        return score.clamp(0.0, 1.0)


class SubgraphContrastive(nn.Module):
    """
    子图级对比学习
    
    对比良性子图和恶意子图的表示。
    """
    
    def __init__(self,
                 hidden_dim: int,
                 subgraph_size: int = 10,
                 temperature: float = 0.07):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.subgraph_size = subgraph_size
        self.temperature = temperature
        
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def sample_subgraph(self, 
                        g: dgl.DGLGraph, 
                        center_node: int,
                        size: int) -> dgl.DGLGraph:
        """采样以center_node为中心的子图"""
        # BFS采样
        nodes = {center_node}
        frontier = [center_node]
        
        while len(nodes) < size and frontier:
            new_frontier = []
            for node in frontier:
                neighbors = g.successors(node).tolist() + g.predecessors(node).tolist()
                for n in neighbors:
                    if n not in nodes and len(nodes) < size:
                        nodes.add(n)
                        new_frontier.append(n)
            frontier = new_frontier
        
        node_list = list(nodes)
        subg = dgl.node_subgraph(g, node_list)
        return subg
    
    def forward(self,
                encoder: nn.Module,
                g: dgl.DGLGraph,
                features: torch.Tensor,
                cic_scores: torch.Tensor,
                n_samples: int = 5) -> torch.Tensor:
        """
        计算子图对比损失
        
        Args:
            encoder: 编码器
            g: 原始图
            features: 节点特征
            cic_scores: CIC分数
            n_samples: 每类采样的子图数量
        """
        device = features.device
        if cic_scores.dim() == 2:
            cic_scores_clean = torch.nan_to_num(cic_scores.to(device), nan=0.0, posinf=1.0, neginf=0.0)
            total_scores = 1.0 - (1.0 - 0.25 * cic_scores_clean).prod(dim=1)
        else:
            total_scores = cic_scores
        total_scores = torch.nan_to_num(total_scores.to(device), nan=0.0, posinf=1.0, neginf=0.0)
        
        # 选择正常和异常中心
        _, high_indices = torch.topk(total_scores, min(n_samples, g.num_nodes()))
        _, low_indices = torch.topk(-total_scores, min(n_samples, g.num_nodes()))
        
        # 采样子图并编码
        benign_reprs = []
        malicious_reprs = []
        
        for idx in low_indices.tolist()[:n_samples]:
            subg = self.sample_subgraph(g, idx, self.subgraph_size)
            sub_features = features[subg.ndata[dgl.NID]]
            embed = encoder(subg, sub_features)
            if isinstance(embed, tuple):
                embed = embed[0]
            repr_ = embed.mean(dim=0)
            benign_reprs.append(repr_)
        
        for idx in high_indices.tolist()[:n_samples]:
            subg = self.sample_subgraph(g, idx, self.subgraph_size)
            sub_features = features[subg.ndata[dgl.NID]]
            embed = encoder(subg, sub_features)
            if isinstance(embed, tuple):
                embed = embed[0]
            repr_ = embed.mean(dim=0)
            malicious_reprs.append(repr_)
        
        if not benign_reprs or not malicious_reprs:
            return torch.tensor(0.0, device=features.device)
        
        benign = torch.stack(benign_reprs)  # [n, hidden]
        malicious = torch.stack(malicious_reprs)
        
        # 投影
        benign_proj = F.normalize(self.projector(benign), dim=-1)
        malicious_proj = F.normalize(self.projector(malicious), dim=-1)
        
        # 良性子图之间应相似
        if benign_proj.size(0) < 2 or malicious_proj.size(0) < 1:
            return torch.tensor(0.0, device=features.device)

        # Supervised contrastive for subgraphs: benign anchors, benign positives, malicious negatives
        sim_bb = torch.mm(benign_proj, benign_proj.t()) / self.temperature
        diag = torch.eye(sim_bb.size(0), device=sim_bb.device, dtype=torch.bool)
        sim_bb = sim_bb.masked_fill(diag, float('-inf'))

        sim_bm = torch.mm(benign_proj, malicious_proj.t()) / self.temperature

        log_pos = torch.logsumexp(sim_bb, dim=1)
        log_den = torch.logsumexp(torch.cat([sim_bb, sim_bm], dim=1), dim=1)
        return -(log_pos - log_den).mean()


# ============================================================================
# 组合模块
# ============================================================================

class CombinedContrastiveLoss(nn.Module):
    """
    组合多种对比损失
    """
    
    def __init__(self,
                 hidden_dim: int,
                 graph_weight: float = 1.0,
                 node_weight: float = 0.5,
                 subgraph_weight: float = 0.5):
        super().__init__()
        
        self.graph_contrastive = CICContrastiveLearning(hidden_dim)
        self.node_contrastive = NodeLevelContrastive(hidden_dim)
        self.subgraph_contrastive = SubgraphContrastive(hidden_dim)
        
        self.graph_weight = graph_weight
        self.node_weight = node_weight
        self.subgraph_weight = subgraph_weight
    
    def forward(self,
                encoder: nn.Module,
                g: dgl.DGLGraph,
                features: torch.Tensor,
                node_embeddings: torch.Tensor,
                cic_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算组合对比损失
        """
        losses = {}
        
        # 图级对比
        graph_loss = self.graph_contrastive(encoder, g, features, cic_scores)
        losses['graph'] = graph_loss * self.graph_weight
        
        # 节点级对比
        node_loss = self.node_contrastive(node_embeddings, cic_scores)
        losses['node'] = node_loss * self.node_weight
        
        # 子图级对比
        subgraph_loss = self.subgraph_contrastive(encoder, g, features, cic_scores)
        losses['subgraph'] = subgraph_loss * self.subgraph_weight
        
        # 总损失
        losses['total'] = sum(losses.values())
        
        return losses


# ============================================================================
# 工厂函数
# ============================================================================

def create_contrastive_module(hidden_dim: int,
                               level: str = 'combined',
                               **kwargs) -> nn.Module:
    """
    创建对比学习模块
    
    Args:
        hidden_dim: 隐藏维度
        level: 对比级别 ('graph', 'node', 'subgraph', 'combined')
        **kwargs: 其他参数
    """
    if level == 'graph':
        return CICContrastiveLearning(hidden_dim, **kwargs)
    elif level == 'node':
        return NodeLevelContrastive(hidden_dim, **kwargs)
    elif level == 'subgraph':
        return SubgraphContrastive(hidden_dim, **kwargs)
    elif level == 'combined':
        return CombinedContrastiveLoss(hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unknown contrastive level: {level}")
