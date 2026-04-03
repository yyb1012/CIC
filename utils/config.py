import argparse


def build_args():
    parser = argparse.ArgumentParser(description="MAGIC")
    
    # ============================================================================
    # 【常用参数】每次运行前请根据需要修改
    # ============================================================================
    
    # 数据集名称: wget, streamspot, theia, cadets, trace
    parser.add_argument("--dataset", type=str, default="theia",
                        help="数据集名称")
    
    # GPU 设备编号: 0, 1, ... (-1 表示 CPU，但训练必须用 GPU)
    parser.add_argument("--device", type=int, default=-1,
                        help="GPU 设备编号，-1 表示 CPU")
    
    # ============================================================================
    # 【训练参数】一般使用默认值，可根据实验调整
    # ============================================================================
    
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="权重衰减 (L2 正则化)")
    parser.add_argument("--optimizer", type=str, default="adamW",
                        help="优化器类型: adam, sgd, adagrad")
    parser.add_argument("--loss_fn", type=str, default='sce',
                        help="损失函数: sce, mse")
    parser.add_argument("--mask_rate", type=float, default=0.5,
                        help="掩码比例 (MAE 用)")
    parser.add_argument("--alpha_l", type=float, default=3,
                        help="`pow` index for `sce` loss")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="GAT 中 LeakyReLU 的负斜率")
    parser.add_argument("--pooling", type=str, default="mean",
                        help="图池化方式: mean, sum, max")
    parser.add_argument("--max_epoch", type=int, default=300,
                        help="Max training epochs; 0 uses dataset defaults")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for batch-level datasets; 0 uses defaults")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of evaluation seeds to run")
    parser.add_argument("--seed_start", type=int, default=0,
                        help="Start seed value for evaluation runs")
    
    # ============================================================================
    # 【早停参数】用于防止过拟合，启用后会从训练集中划分验证集
    # ============================================================================
    
    # 是否启用早停 (命令行加 --early_stop 即可启用)
    parser.add_argument("--early_stop", action="store_true", default=True,
                        help="启用基于验证集损失的早停机制")
    
    # 验证集比例 (从训练集中划分)
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集占训练集的比例 (0.0~1.0)")
    
    # 早停耐心值：验证损失连续 N 个 epoch 无改善则停止
    parser.add_argument("--early_stop_patience", type=int, default=5,
                        help="验证损失无改善的最大轮数")
    
    # 最小改善阈值：验证损失改善需超过此值才视为有效
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                        help="判定为改善的最小损失下降量")
    parser.add_argument("--val_every_n", type=int, default=4,
                        help="Validate every N epochs when early stopping is enabled")
    
    # ========== CIC 不变量相关参数 ==========
    # 时间常数（纳秒级）
    parser.add_argument("--cic_time_constant", type=float, default=1e9,
                        help="Time constant for reach violation decay (nanoseconds)")
    parser.add_argument("--cic_timing_lambda", type=float, default=1e-9,
                        help="Lambda for timing violation exponential decay")
    parser.add_argument("--cic_alias_lambda", type=float, default=0.5,
                        help="Lambda for alias violation exponential decay")
    
    # 不变量融合权重 (I_reach, I_creator, I_timing, I_alias)
    parser.add_argument("--cic_weight_reach", type=float, default=0.25,
                        help="Weight for reachability invariant")
    parser.add_argument("--cic_weight_creator", type=float, default=0.25,
                        help="Weight for creator consistency invariant")
    parser.add_argument("--cic_weight_timing", type=float, default=0.25,
                        help="Weight for timing invariant")
    parser.add_argument("--cic_weight_alias", type=float, default=0.25,
                        help="Weight for alias invariant")
    
    # CIC 功能开关：默认启用（项目内必用）；如需做对照实验，显式传 --no_cic
    parser.set_defaults(use_cic=True)
    parser.add_argument("--no_cic", dest="use_cic", action="store_false",
                        help="Disable CIC invariant features (for ablation)")
    # 兼容旧参数（隐藏，不建议使用）
    parser.add_argument("--use_cic", dest="use_cic", action="store_true",
                        help=argparse.SUPPRESS)

    # 默认把 CIC 分数拼到节点特征；如需关闭用于对照实验，显式传 --no_cic_as_node_feature
    parser.set_defaults(cic_as_node_feature=True)
    parser.add_argument("--no_cic_as_node_feature", dest="cic_as_node_feature", action="store_false",
                        help="Do not append CIC scores to node features (for ablation)")
    # 兼容旧参数（隐藏，不建议使用）
    parser.add_argument("--cic_as_node_feature", dest="cic_as_node_feature", action="store_true",
                        help=argparse.SUPPRESS)
    
    # 异常检测阈值
    parser.add_argument("--cic_anomaly_threshold", type=float, default=0.5,
                        help="Threshold for anomaly candidate detection")
    
    # ========== Phase 3: 单调融合 + 排序一致性 ==========
    parser.add_argument("--fusion_type", type=str, default="risk_amplification",
                        choices=["weighted_sum", "risk_amplification", "max"],
                        help="Invariant fusion type")
    parser.add_argument("--ranking_margin", type=float, default=1.0,
                        help="Margin for ranking consistency loss")
    parser.add_argument("--ranking_max_pairs", type=int, default=10000,
                        help="Max node pairs to sample for ranking loss")
    
    # 组合损失权重
    parser.add_argument("--recon_loss_weight", type=float, default=1.0,
                        help="Weight for reconstruction loss")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.5,
                        help="Weight for contrastive loss")
    parser.add_argument("--ranking_loss_weight", type=float, default=0.3,
                        help="Weight for ranking consistency loss")
    parser.add_argument("--learnable_loss_weights", action="store_true", default=False,
                        help="Use learnable loss weights")
    
    # ========== Phase 4: 解释子图构建 ==========
    parser.add_argument("--explanation_k_hop", type=int, default=2,
                        help="K-hop radius for explanation subgraph")
    parser.add_argument("--explanation_threshold", type=float, default=0.0,
                        help="Edge importance threshold for explanation")
    parser.add_argument("--explanation_top_k", type=int, default=10,
                        help="Top-k anomaly nodes to explain")
    # 默认保存中间值；如需关闭，显式传 --no_save_intermediate
    parser.set_defaults(save_intermediate=True)
    parser.add_argument("--no_save_intermediate", dest="save_intermediate", action="store_false",
                        help="Do not save intermediate values during training")
    # 兼容旧参数（隐藏，不建议使用）
    parser.add_argument("--save_intermediate", dest="save_intermediate", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--intermediate_save_dir", type=str, default="./checkpoints",
                        help="Directory to save intermediate values")
    parser.add_argument("--explanation_output_dir", type=str, default="./explanation_graphs",
                        help="Directory for explanation graph output")
    
    args = parser.parse_args()
    return args


def get_cic_config(args):
    """从args中提取CIC相关配置"""
    return {
        'time_constant': args.cic_time_constant,
        'timing_lambda': args.cic_timing_lambda,
        'alias_lambda': args.cic_alias_lambda,
        'weights': [
            args.cic_weight_reach,
            args.cic_weight_creator,
            args.cic_weight_timing,
            args.cic_weight_alias
        ],
        'use_cic': args.use_cic,
        'as_node_feature': args.cic_as_node_feature,
        'anomaly_threshold': args.cic_anomaly_threshold
    }


def get_fusion_config(args):
    """从args中提取融合相关配置"""
    return {
        'fusion_type': args.fusion_type,
        'ranking_margin': args.ranking_margin,
        'ranking_max_pairs': args.ranking_max_pairs,
        'recon_weight': args.recon_loss_weight,
        'contrastive_weight': args.contrastive_loss_weight,
        'ranking_weight': args.ranking_loss_weight,
        'learnable_weights': args.learnable_loss_weights
    }


def get_explanation_config(args):
    """从args中提取解释相关配置"""
    return {
        'k_hop': args.explanation_k_hop,
        'threshold': args.explanation_threshold,
        'top_k': args.explanation_top_k,
        'save_intermediate': args.save_intermediate,
        'save_dir': args.intermediate_save_dir,
        'output_dir': args.explanation_output_dir
    }
