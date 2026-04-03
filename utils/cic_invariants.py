"""
CIC (Causal Invariant Consistency) 不变量计算模块

基于研究路线实现四种不变量的计算：
- I_reach: 执行者对文件的可达性违例
- I_creator: 创建者/首写者一致性违例  
- I_timing: 创建→写→执行的时序性违例
- I_alias: 多别名/同形字符违例

参考: FIELD_DOCUMENTATION.md
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np


# ============================================================================
# 数据结构定义（与trace_parser.py保持一致）
# ============================================================================

@dataclass
class AccessRecord:
    """访问记录"""
    subject_uuid: str
    object_uuid: str
    access_type: str  # read, write, exec, mmap, create
    timestamp: int
    subject_uid: str = ""
    subject_mnt_ns: str = ""


@dataclass
class InvariantScores:
    """单个实体的不变量分数"""
    entity_uuid: str
    i_reach: float = 0.0      # 可达性违例分数 [0, 1]
    i_creator: float = 0.0    # 创建者一致性违例分数 [0, 1]
    i_timing: float = 0.0     # 时序违例分数 [0, 1]
    i_alias: float = 0.0      # 别名违例分数 [0, 1]
    
    def to_vector(self) -> np.ndarray:
        """转换为4维向量"""
        return np.array([self.i_reach, self.i_creator, self.i_timing, self.i_alias])
    
    def total_score(self, weights: Optional[List[float]] = None) -> float:
        """
        计算CIC总分（风险放大聚合）

        研究路线给出的形式：S(e) = 1 - Π_k (1 - w_k * v_k(e))
        其中 v_k(e)∈[0,1] 是每条不变量的违例强度。
        """
        if weights is None:
            weights = [0.25, 0.25, 0.25, 0.25]
        prod = 1.0
        for w, v in zip(weights, self.to_vector()):
            prod *= (1.0 - float(w) * float(v))
        prod = max(0.0, min(1.0, prod))
        return 1.0 - prod


# ============================================================================
# CIC 不变量计算器
# ============================================================================

class CICInvariantComputer:
    """
    CIC不变量计算器
    
    用于计算图中每个节点的不变量违例分数。
    分数越高表示越可能是异常。
    """
    
    # 默认超参数（可通过config覆盖）
    DEFAULT_TIME_CONSTANT = 1e9  # 1秒（纳秒级）
    DEFAULT_TIMING_LAMBDA = 1e-9
    DEFAULT_ALIAS_LAMBDA = 0.5
    DEFAULT_WEIGHTS = [0.25, 0.25, 0.25, 0.25]  # I_reach, I_creator, I_timing, I_alias
    
    def __init__(self, 
                 subjects: Dict[str, Any],
                 file_objects: Dict[str, Any],
                 netflow_objects: Dict[str, Any],
                 memory_objects: Dict[str, Any],
                 file_access_history: Dict[str, List[AccessRecord]],
                 subject_access_history: Dict[str, List[AccessRecord]],
                 inode_to_files: Dict[str, List[str]],
                 config: Optional[Dict] = None):
        """
        初始化CIC计算器
        
        Args:
            subjects: UUID -> SubjectInfo 映射
            file_objects: UUID -> FileObjectInfo 映射
            netflow_objects: UUID -> NetFlowObjectInfo 映射
            memory_objects: UUID -> MemoryObjectInfo 映射
            file_access_history: file_uuid -> [AccessRecord] 映射
            subject_access_history: subject_uuid -> [AccessRecord] 映射
            inode_to_files: "dev:inode" -> [file_uuid] 映射
            config: 超参数配置字典 (可选)
        """
        self.subjects = subjects
        self.file_objects = file_objects
        self.netflow_objects = netflow_objects
        self.memory_objects = memory_objects
        self.file_access_history = file_access_history
        self.subject_access_history = subject_access_history
        self.inode_to_files = inode_to_files
        
        # 设置超参数
        self.TIME_CONSTANT = self.DEFAULT_TIME_CONSTANT
        self.TIMING_LAMBDA = self.DEFAULT_TIMING_LAMBDA
        self.ALIAS_LAMBDA = self.DEFAULT_ALIAS_LAMBDA
        self.WEIGHTS = self.DEFAULT_WEIGHTS.copy()
        
        if config:
            self.set_config(config)
        
        # 缓存计算结果
        self._score_cache: Dict[str, InvariantScores] = {}
    
    def set_config(self, config: Dict):
        """设置超参数配置"""
        if 'time_constant' in config:
            self.TIME_CONSTANT = config['time_constant']
        if 'timing_lambda' in config:
            self.TIMING_LAMBDA = config['timing_lambda']
        if 'alias_lambda' in config:
            self.ALIAS_LAMBDA = config['alias_lambda']
        if 'weights' in config:
            self.WEIGHTS = config['weights']
    
    # ========================================================================
    # I_reach: 可达性不变量
    # ========================================================================
    
    def compute_reach_violation(self, 
                                 subject_uuid: str, 
                                 file_uuid: str, 
                                 exec_timestamp: int,
                                 expected_mnt_ns: Optional[str] = None) -> float:
        """
        计算可达性违例强度
        
        定义：进程执行文件前，应该对其有读/映射路径；凭空执行具有可疑性。
        
        Args:
            subject_uuid: 执行进程的UUID
            file_uuid: 被执行文件的UUID  
            exec_timestamp: 执行时间戳
            expected_mnt_ns: 期望的挂载命名空间
            
        Returns:
            违例分数 [0, 1]，越高表示越异常
        """
        history = self.file_access_history.get(file_uuid, [])

        # 检查执行前是否有读/映射记录（避免构建大列表）
        last_access_time = None
        for r in history:
            if r.subject_uuid != subject_uuid:
                continue
            if r.access_type not in ('read', 'mmap'):
                continue
            if r.timestamp >= exec_timestamp:
                continue
            if last_access_time is None or r.timestamp > last_access_time:
                last_access_time = r.timestamp

        if last_access_time is None:
            return 1.0  # 严重违例：凭空执行

        # 时间衰减：距离上次访问越久，分数越高
        time_gap = exec_timestamp - last_access_time
        decay = 1.0 - math.exp(-time_gap / self.TIME_CONSTANT)
        
        # 命名空间检查
        ns_penalty = 0.0
        if expected_mnt_ns:
            subject = self.subjects.get(subject_uuid)
            if subject:
                subject_mnt_ns = getattr(subject, 'mnt_ns', None) or subject.get('mnt_ns', '')
                if subject_mnt_ns and subject_mnt_ns != expected_mnt_ns:
                    ns_penalty = 0.2
        
        return min(1.0, decay + ns_penalty)
    
    def compute_reach_score_for_subject(self, subject_uuid: str) -> float:
        """计算进程的整体可达性违例分数"""
        history = self.subject_access_history.get(subject_uuid, [])

        if not history:
            return 0.0

        # 为保证正确性，按时间排序后单遍扫描：使用进程自身 read/mmap 轨迹计算 reach
        sorted_hist = sorted(history, key=lambda r: r.timestamp)
        last_read_or_mmap: Dict[str, Tuple[int, str]] = {}
        max_violation = 0.0

        for r in sorted_hist:
            if r.access_type in ('read', 'mmap'):
                last_read_or_mmap[r.object_uuid] = (r.timestamp, r.subject_mnt_ns)
                continue

            if r.access_type != 'exec':
                continue

            prev = last_read_or_mmap.get(r.object_uuid)
            if not prev:
                max_violation = max(max_violation, 1.0)
                continue

            last_ts, last_mnt_ns = prev
            time_gap = r.timestamp - last_ts
            if time_gap < 0:
                max_violation = max(max_violation, 1.0)
                continue

            decay = 1.0 - math.exp(-time_gap / self.TIME_CONSTANT)
            ns_penalty = 0.0
            if last_mnt_ns and r.subject_mnt_ns and last_mnt_ns != r.subject_mnt_ns:
                ns_penalty = 0.2
            max_violation = max(max_violation, min(1.0, decay + ns_penalty))

        return max_violation
    
    # ========================================================================
    # I_creator: 创建者一致性不变量
    # ========================================================================
    
    def compute_creator_violation(self, file_uuid: str) -> float:
        """
        计算创建者一致性违例强度
        
        定义：文件的创建者与首写者应具有较高一致性。
        
        Args:
            file_uuid: 文件UUID
            
        Returns:
            违例分数 [0, 1]，越高表示越异常
        """
        fobj = self.file_objects.get(file_uuid)
        if not fobj:
            return 0.0
        
        # 兼容dict和dataclass
        def get_attr(obj, attr, default=""):
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return getattr(obj, attr, default)
        
        creator_uuid = get_attr(fobj, 'creator_uuid', '')
        first_writer_uuid = get_attr(fobj, 'first_writer_uuid', '')
        
        # 如果没有创建者或首写者信息
        if not creator_uuid or not first_writer_uuid:
            return 0.0
        
        # 如果创建者和首写者相同
        if creator_uuid == first_writer_uuid:
            return 0.0
        
        # 计算相似度
        creator_uid = get_attr(fobj, 'creator_uid', '')
        creator_gid = get_attr(fobj, 'creator_gid', '')
        creator_mnt_ns = get_attr(fobj, 'creator_mnt_ns', '')
        
        first_writer_uid = get_attr(fobj, 'first_writer_uid', '')
        first_writer_gid = get_attr(fobj, 'first_writer_gid', '')
        first_writer_mnt_ns = get_attr(fobj, 'first_writer_mnt_ns', '')
        
        uid_match = 1.0 if creator_uid == first_writer_uid else 0.0
        gid_match = 1.0 if creator_gid == first_writer_gid else 0.0
        ns_match = 1.0 if creator_mnt_ns == first_writer_mnt_ns else 0.0
        
        similarity = (uid_match + gid_match + ns_match) / 3.0
        return 1.0 - similarity
    
    # ========================================================================
    # I_timing: 时序不变量
    # ========================================================================
    
    def compute_timing_violation(self, 
                                  file_uuid: str, 
                                  exec_timestamp: int) -> float:
        """
        计算时序违例强度
        
        定义：一个文件被执行前，必须先被创建且至少一次写入。
        时间越短，分数越高（可能是写入即执行的恶意行为）。
        
        Args:
            file_uuid: 文件UUID
            exec_timestamp: 执行时间戳
            
        Returns:
            违例分数 [0, 1]
        """
        history = self.file_access_history.get(file_uuid, [])

        # 只考虑执行前的 create/write，避免把“执行之后的写入”误判为时序逆序
        last_write_time = None
        has_create = False
        for r in history:
            if r.timestamp >= exec_timestamp:
                continue
            if r.access_type == 'write':
                if last_write_time is None or r.timestamp > last_write_time:
                    last_write_time = r.timestamp
            elif r.access_type == 'create':
                has_create = True

        if last_write_time is None:
            v = 0.7  # 未写入就执行：较强违例（兼容系统文件等场景，不直接给 1.0）
            if not has_create:
                v = min(1.0, v + 0.2)
            return v

        delta_t = exec_timestamp - last_write_time
        if delta_t < 0:
            return 1.0

        v = math.exp(-self.TIMING_LAMBDA * delta_t)  # 时间越短，违例越高
        if not has_create:
            v = min(1.0, v + 0.2)
        return v
    
    def compute_timing_score_for_file(self, file_uuid: str) -> float:
        """计算文件的整体时序违例分数"""
        history = self.file_access_history.get(file_uuid, [])

        if not history:
            return 0.0

        # 为保证正确性，按时间排序后单遍扫描：维护 last_write / create 标记
        sorted_hist = sorted(history, key=lambda r: r.timestamp)
        last_write_time = None
        has_create = False
        max_violation = 0.0

        for r in sorted_hist:
            if r.access_type == 'create':
                has_create = True
                continue
            if r.access_type == 'write':
                last_write_time = r.timestamp if last_write_time is None else max(last_write_time, r.timestamp)
                continue
            if r.access_type != 'exec':
                continue

            if last_write_time is None:
                v = 0.7 + (0.2 if not has_create else 0.0)
                max_violation = max(max_violation, min(1.0, v))
                continue

            delta_t = r.timestamp - last_write_time
            if delta_t < 0:
                max_violation = max(max_violation, 1.0)
                continue
            v = math.exp(-self.TIMING_LAMBDA * delta_t)
            if not has_create:
                v = min(1.0, v + 0.2)
            max_violation = max(max_violation, v)

        return max_violation
    
    # ========================================================================
    # I_alias: 多别名不变量
    # ========================================================================
    
    def compute_alias_violation(self, file_uuid: str) -> float:
        """
        计算别名违例强度
        
        定义：同一inode在短时间内出现异常数量的别名。
        
        Args:
            file_uuid: 文件UUID
            
        Returns:
            违例分数 [0, 1]
        """
        fobj = self.file_objects.get(file_uuid)
        if not fobj:
            return 0.0
        
        # 兼容dict和dataclass
        def get_attr(obj, attr, default=""):
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return getattr(obj, attr, default)
        
        inode = get_attr(fobj, 'inode', '')
        dev = get_attr(fobj, 'dev', '')
        
        if not inode:
            return 0.0
        
        key = f"{dev}:{inode}"
        alias_files = self.inode_to_files.get(key, [])
        alias_count = len(alias_files)
        
        # 单个别名是正常的
        if alias_count <= 1:
            return 0.0
        
        # 使用指数函数避免无限增长
        return 1.0 - math.exp(-self.ALIAS_LAMBDA * (alias_count - 1))
    
    # ========================================================================
    # 综合计算
    # ========================================================================
    
    def compute_scores_for_file(self, file_uuid: str) -> InvariantScores:
        """计算文件节点的所有不变量分数"""
        if file_uuid in self._score_cache:
            return self._score_cache[file_uuid]
        
        scores = InvariantScores(
            entity_uuid=file_uuid,
            i_reach=0.0,  # 文件节点不直接计算可达性
            i_creator=self.compute_creator_violation(file_uuid),
            i_timing=self.compute_timing_score_for_file(file_uuid),
            i_alias=self.compute_alias_violation(file_uuid)
        )
        
        self._score_cache[file_uuid] = scores
        return scores
    
    def compute_scores_for_subject(self, subject_uuid: str) -> InvariantScores:
        """计算进程节点的所有不变量分数"""
        if subject_uuid in self._score_cache:
            return self._score_cache[subject_uuid]
        
        scores = InvariantScores(
            entity_uuid=subject_uuid,
            i_reach=self.compute_reach_score_for_subject(subject_uuid),
            i_creator=0.0,  # 进程节点不直接计算创建者一致性
            i_timing=0.0,   # 进程节点的时序在边上计算
            i_alias=0.0     # 进程节点不计算别名
        )
        
        self._score_cache[subject_uuid] = scores
        return scores
    
    def compute_scores_for_entity(self, entity_uuid: str) -> InvariantScores:
        """计算任意实体的不变量分数"""
        if entity_uuid in self._score_cache:
            return self._score_cache[entity_uuid]
        
        if entity_uuid in self.subjects:
            return self.compute_scores_for_subject(entity_uuid)
        elif entity_uuid in self.file_objects:
            return self.compute_scores_for_file(entity_uuid)
        else:
            # NetFlowObject, MemoryObject 等暂不计算
            return InvariantScores(entity_uuid=entity_uuid)
    
    def compute_all_scores(self) -> Dict[str, InvariantScores]:
        """计算所有实体的不变量分数"""
        all_scores = {}
        
        # 计算所有进程的分数
        for uuid in self.subjects:
            all_scores[uuid] = self.compute_scores_for_subject(uuid)
        
        # 计算所有文件的分数
        for uuid in self.file_objects:
            all_scores[uuid] = self.compute_scores_for_file(uuid)
        
        return all_scores
    
    def get_anomaly_candidates(self, 
                                threshold: float = 0.5,
                                top_k: Optional[int] = None) -> List[Tuple[str, InvariantScores]]:
        """
        获取异常候选实体
        
        Args:
            threshold: 总分阈值
            top_k: 返回前k个最高分的实体
            
        Returns:
            (entity_uuid, scores) 列表，按总分降序
        """
        all_scores = self.compute_all_scores()
        
        # 按总分排序
        sorted_scores = sorted(
            all_scores.items(),
            key=lambda x: x[1].total_score(),
            reverse=True
        )
        
        # 过滤和截断
        if threshold is not None:
            sorted_scores = [(k, v) for k, v in sorted_scores if v.total_score() >= threshold]
        
        if top_k is not None:
            sorted_scores = sorted_scores[:top_k]
        
        return sorted_scores
    
    def clear_cache(self):
        """清除缓存"""
        self._score_cache.clear()


# ============================================================================
# 边级不变量计算
# ============================================================================

class EdgeInvariantComputer:
    """
    边级CIC不变量计算器
    
    用于计算图中每条边的不变量违例分数。
    """
    
    def __init__(self, node_computer: CICInvariantComputer):
        self.node_computer = node_computer
    
    def compute_edge_reach_violation(self, 
                                      src_uuid: str, 
                                      dst_uuid: str,
                                      edge_type: str,
                                      timestamp: int) -> float:
        """计算边的可达性违例"""
        # 只对执行类边计算
        if 'EXEC' not in edge_type and 'LOAD' not in edge_type:
            return 0.0
        
        return self.node_computer.compute_reach_violation(
            src_uuid, dst_uuid, timestamp
        )
    
    def compute_edge_timing_violation(self,
                                       dst_uuid: str,
                                       edge_type: str,
                                       timestamp: int) -> float:
        """计算边的时序违例"""
        # 只对执行类边计算
        if 'EXEC' not in edge_type and 'LOAD' not in edge_type:
            return 0.0
        
        return self.node_computer.compute_timing_violation(dst_uuid, timestamp)
    
    def compute_edge_scores(self,
                            src_uuid: str,
                            dst_uuid: str, 
                            edge_type: str,
                            timestamp: int) -> Dict[str, float]:
        """计算边的所有不变量分数"""
        return {
            'i_reach': self.compute_edge_reach_violation(src_uuid, dst_uuid, edge_type, timestamp),
            'i_timing': self.compute_edge_timing_violation(dst_uuid, edge_type, timestamp),
        }


# ============================================================================
# 工具函数
# ============================================================================

def create_invariant_computer_from_state(state) -> CICInvariantComputer:
    """
    从trace_parser的state对象创建CIC计算器
    
    Args:
        state: ParserState对象
        
    Returns:
        CICInvariantComputer实例
    """
    return CICInvariantComputer(
        subjects=state.subjects,
        file_objects=state.file_objects,
        netflow_objects=state.netflow_objects,
        memory_objects=state.memory_objects,
        file_access_history=dict(state.file_access_history),
        subject_access_history=dict(state.subject_access_history),
        inode_to_files=dict(state.inode_to_files)
    )


def create_invariant_computer_from_pkl(data_dir: str) -> CICInvariantComputer:
    """
    从PKL文件创建CIC计算器（内存优化版）
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        CICInvariantComputer实例
    """
    import pickle as pkl
    import os
    import gc
    
    # 加载entities.pkl
    entities_path = os.path.join(data_dir, 'entities.pkl')
    if not os.path.exists(entities_path):
        raise FileNotFoundError(f"找不到entities.pkl: {entities_path}")
    
    print(f"[CIC] 加载 entities.pkl...")
    with open(entities_path, 'rb') as f:
        entities = pkl.load(f)
    
    # 提取必要的字段并立即创建浅拷贝引用
    subjects = entities.get('subjects', {})
    file_objects = entities.get('file_objects', {})
    netflow_objects = entities.get('netflow_objects', {})
    memory_objects = entities.get('memory_objects', {})
    
    # 释放 entities 字典本身（子字典已被引用）
    del entities
    gc.collect()
    
    # 加载invariant_tracking.pkl
    invariant_path = os.path.join(data_dir, 'invariant_tracking.pkl')
    if not os.path.exists(invariant_path):
        raise FileNotFoundError(f"找不到invariant_tracking.pkl: {invariant_path}")
    
    print(f"[CIC] 加载 invariant_tracking.pkl...")
    with open(invariant_path, 'rb') as f:
        invariant_data = pkl.load(f)
    
    # 提取 inode_to_files
    inode_to_files = invariant_data.get('inode_to_files', {})
    
    # 增量重建AccessRecord对象 - 分批处理以减少峰值内存
    print(f"[CIC] 构建 file_access_history...")
    file_access_history = defaultdict(list)
    raw_file_history = invariant_data.get('file_access_history', {})
    batch_keys = list(raw_file_history.keys())
    batch_size = 10000
    def to_access_record(item):
        if isinstance(item, AccessRecord):
            return item
        if isinstance(item, dict):
            return AccessRecord(**item)
        return AccessRecord(**dict(item))
    
    for i in range(0, len(batch_keys), batch_size):
        batch = batch_keys[i:i+batch_size]
        for k in batch:
            v = raw_file_history[k]
            file_access_history[k] = [to_access_record(r) for r in v]
        # 每批处理后触发垃圾回收
        if i > 0 and i % (batch_size * 10) == 0:
            gc.collect()
    
    # 释放原始数据
    del raw_file_history
    gc.collect()
    
    print(f"[CIC] 构建 subject_access_history...")
    subject_access_history = defaultdict(list)
    raw_subject_history = invariant_data.get('subject_access_history', {})
    batch_keys = list(raw_subject_history.keys())
    
    for i in range(0, len(batch_keys), batch_size):
        batch = batch_keys[i:i+batch_size]
        for k in batch:
            v = raw_subject_history[k]
            subject_access_history[k] = [to_access_record(r) for r in v]
        if i > 0 and i % (batch_size * 10) == 0:
            gc.collect()
    
    # 释放原始数据和invariant_data
    del raw_subject_history
    del invariant_data
    gc.collect()
    
    print(f"[CIC] CIC计算器构建完成")
    return CICInvariantComputer(
        subjects=subjects,
        file_objects=file_objects,
        netflow_objects=netflow_objects,
        memory_objects=memory_objects,
        file_access_history=file_access_history,
        subject_access_history=subject_access_history,
        inode_to_files=inode_to_files
    )


def save_cic_scores(scores: Dict[str, InvariantScores], data_dir: str, filename: str = 'cic_scores.pkl'):
    """
    保存CIC分数到PKL文件
    
    Args:
        scores: uuid -> InvariantScores 的字典
        data_dir: 数据目录路径
        filename: 保存的文件名
    """
    import pickle as pkl
    import os
    
    # 转换为可序列化的格式
    serializable_scores = {}
    for uuid, score in scores.items():
        serializable_scores[uuid] = {
            'entity_uuid': score.entity_uuid,
            'i_reach': score.i_reach,
            'i_creator': score.i_creator,
            'i_timing': score.i_timing,
            'i_alias': score.i_alias,
            'vector': score.to_vector().tolist(),
            'total': score.total_score()
        }
    
    save_path = os.path.join(data_dir, filename)
    with open(save_path, 'wb') as f:
        pkl.dump(serializable_scores, f)
    
    print(f"[CIC] 已保存 {len(scores)} 个实体的CIC分数到 {filename}")
    return save_path


def load_cic_scores(data_dir: str, filename: str = 'cic_scores.pkl') -> Dict[str, InvariantScores]:
    """
    从PKL文件加载CIC分数
    
    Args:
        data_dir: 数据目录路径
        filename: 文件名
        
    Returns:
        uuid -> InvariantScores 的字典
    """
    import pickle as pkl
    import os
    
    load_path = os.path.join(data_dir, filename)
    if not os.path.exists(load_path):
        return {}
    
    with open(load_path, 'rb') as f:
        serializable_scores = pkl.load(f)
    
    # 重建InvariantScores对象
    scores = {}
    for uuid, data in serializable_scores.items():
        scores[uuid] = InvariantScores(
            entity_uuid=data['entity_uuid'],
            i_reach=data['i_reach'],
            i_creator=data['i_creator'],
            i_timing=data['i_timing'],
            i_alias=data['i_alias']
        )
    
    print(f"[CIC] 已加载 {len(scores)} 个实体的CIC分数")
    return scores


def compute_and_save_cic_scores(data_dir: str, config: Optional[Dict] = None) -> Dict[str, InvariantScores]:
    """
    计算CIC分数并保存到文件
    
    Args:
        data_dir: 数据目录路径
        config: 超参数配置
        
    Returns:
        uuid -> InvariantScores 的字典
    """
    # 检查是否已有缓存的分数
    import os
    cache_path = os.path.join(data_dir, 'cic_scores.pkl')
    if os.path.exists(cache_path):
        scores = load_cic_scores(data_dir)
        if scores:
            return scores
    
    # 创建计算器并计算
    computer = create_invariant_computer_from_pkl(data_dir)
    if config:
        computer.set_config(config)
    
    scores = computer.compute_all_scores()
    
    # 保存分数
    save_cic_scores(scores, data_dir)
    
    return scores


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CIC Invariant Computer Test')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory path')
    args = parser.parse_args()
    
    print(f"加载数据目录: {args.data_dir}")
    
    try:
        computer = create_invariant_computer_from_pkl(args.data_dir)
        print(f"✓ 成功创建CIC计算器")
        print(f"  - Subjects: {len(computer.subjects)}")
        print(f"  - FileObjects: {len(computer.file_objects)}")
        print(f"  - FileAccessHistory: {sum(len(v) for v in computer.file_access_history.values())} records")
        
        # 计算所有分数
        print("\n计算不变量分数...")
        all_scores = computer.compute_all_scores()
        print(f"✓ 计算完成，共 {len(all_scores)} 个实体")
        
        # 显示异常候选
        print("\n异常候选（前10个）:")
        candidates = computer.get_anomaly_candidates(threshold=0.3, top_k=10)
        for uuid, scores in candidates:
            print(f"  {uuid[:16]}... | reach={scores.i_reach:.3f} creator={scores.i_creator:.3f} "
                  f"timing={scores.i_timing:.3f} alias={scores.i_alias:.3f} | total={scores.total_score():.3f}")
    
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("请先运行 trace_parser.py 生成 entities.pkl 和 invariant_tracking.pkl")
