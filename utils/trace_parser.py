"""
Enhanced CDM Trace Parser for CIC (Causal Invariant Consistency) Analysis

本解析器基于研究路线要求，提取以下字段用于：
1. CIC不变量计算（I_reach, I_creator, I_timing, I_alias）
2. 不变量感知的自监督训练（掩码重构、对比学习）
3. 异常检测与溯源子图生成

Author: Enhanced for CIC-based APT Detection
"""

import argparse
import json
import os
import re
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Optional, Tuple, Any, Deque
from tqdm import tqdm
import networkx as nx
import pickle as pkl


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class PrincipalInfo:
    """主体权限信息 - 用于身份不变量"""
    uuid: str
    user_id: str = ""
    username: str = ""
    group_ids: List[str] = field(default_factory=list)
    cred: str = ""  # 凭证信息: euid/egid/suid/sgid/...
    host_id: str = ""
    
    # 解析cred获得详细权限
    euid: str = ""
    egid: str = ""
    suid: str = ""
    sgid: str = ""
    fsuid: str = ""
    fsgid: str = ""


@dataclass
class SubjectInfo:
    """进程/主体信息 - 用于执行者可达性不变量"""
    uuid: str
    subject_type: str = ""  # SUBJECT_PROCESS, SUBJECT_THREAD, etc.
    cid: int = 0  # 进程ID
    ppid: str = ""  # 父进程ID
    tgid: str = ""  # 线程组ID
    parent_subject: str = ""
    host_id: str = ""
    local_principal: str = ""  # 关联的Principal UUID
    start_timestamp: int = 0
    cmdline: str = ""
    exe_path: str = ""  # 可执行文件路径
    
    # 命名空间信息（用于跨命名空间不变量）
    mnt_ns: str = ""  # 挂载命名空间
    pid_ns: str = ""  # 进程命名空间
    net_ns: str = ""  # 网络命名空间
    user_ns: str = ""  # 用户命名空间
    
    # 权限信息（从关联的Principal获取）
    uid: str = ""
    gid: str = ""
    euid: str = ""
    egid: str = ""
    
    # 计算字段
    exe_hash: str = ""  # exe_path的哈希


@dataclass  
class FileObjectInfo:
    """文件对象信息 - 用于创建者一致性不变量"""
    uuid: str
    file_type: str = ""  # FILE_OBJECT_FILE, FILE_OBJECT_BLOCK, etc.
    host_id: str = ""
    permission: str = ""
    epoch: int = 0
    
    # 文件标识（用于I_alias多别名不变量）
    inode: str = ""
    dev: str = ""  # 设备号
    filename: str = ""  # 文件路径
    
    # 关联主体
    local_principal: str = ""
    
    # 不变量追踪字段
    creator_uuid: str = ""  # 创建者进程UUID
    creator_uid: str = ""
    creator_gid: str = ""
    creator_mnt_ns: str = ""
    creator_timestamp: int = 0
    
    first_writer_uuid: str = ""  # 首写者进程UUID
    first_writer_uid: str = ""
    first_writer_gid: str = ""
    first_writer_mnt_ns: str = ""
    first_write_timestamp: int = 0
    
    # 别名信息
    aliases: List[str] = field(default_factory=list)  # 同inode的其他路径


@dataclass
class MemoryObjectInfo:
    """内存对象信息"""
    uuid: str
    host_id: str = ""
    memory_address: int = 0
    size: int = 0
    permission: str = ""


@dataclass
class NetFlowObjectInfo:
    """网络流对象信息"""
    uuid: str
    host_id: str = ""
    local_address: str = ""
    local_port: int = 0
    remote_address: str = ""
    remote_port: int = 0
    ip_protocol: str = ""


@dataclass
class EventInfo:
    """事件信息 - 用于时序不变量"""
    uuid: str
    event_type: str = ""
    sequence: int = 0
    thread_id: int = 0
    host_id: str = ""
    timestamp: int = 0
    
    # 关联实体
    subject_uuid: str = ""  # 源（通常是进程）
    predicate_object_uuid: str = ""  # 目标1
    predicate_object2_uuid: str = ""  # 目标2
    predicate_object_path: str = ""
    predicate_object2_path: str = ""
    
    # 事件属性
    cmdline: str = ""
    flags: str = ""
    mode: str = ""
    prot: str = ""  # 内存保护标志
    return_code: str = ""
    size: int = 0
    
    # 计算字段（用于不变量）
    is_read: bool = False
    is_write: bool = False
    is_exec: bool = False
    is_create: bool = False
    is_mmap: bool = False
    is_connect: bool = False
    is_send: bool = False
    is_recv: bool = False


@dataclass
class AccessRecord:
    """访问记录 - 用于可达性不变量追踪"""
    subject_uuid: str
    object_uuid: str
    access_type: str  # read, write, exec, mmap, create
    timestamp: int
    subject_uid: str = ""
    subject_mnt_ns: str = ""


@dataclass
class EdgeInfo:
    """增强的边信息"""
    src_uuid: str
    dst_uuid: str
    src_type: str
    dst_type: str
    edge_type: str
    timestamp: int
    
    # 边属性
    flags: str = ""
    mode: str = ""
    prot: str = ""
    size: int = 0
    return_code: str = ""
    
    # 源节点属性快照（用于对比学习）
    src_uid: str = ""
    src_gid: str = ""
    src_mnt_ns: str = ""
    src_exe_hash: str = ""
    
    # 目标节点属性快照
    dst_inode: str = ""
    dst_path: str = ""


# ============================================================================
# 正则表达式模式
# ============================================================================

# 基础字段
pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')
pattern_host_id = re.compile(r'hostId\":\"(.*?)\"')

# Event相关
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1_path = re.compile(r'predicateObjectPath\":{\"string\":\"(.*?)\"}')
pattern_dst2_path = re.compile(r'predicateObject2Path\":{\"string\":\"(.*?)\"}')
pattern_sequence = re.compile(r'sequence\":{\"long\":(.*?)}')
pattern_thread_id = re.compile(r'threadId\":{\"int\":(.*?)}')
pattern_size = re.compile(r'size\":{\"long\":(.*?)}')

# Event properties
pattern_cmdline_prop = re.compile(r'\"cmdLine\":\"(.*?)\"')
pattern_flags = re.compile(r'\"flags\":\"(.*?)\"')
pattern_mode = re.compile(r'\"mode\":\"(.*?)\"')
pattern_prot = re.compile(r'\"prot\":\"(.*?)\"')
pattern_rc = re.compile(r'\"rc\":\"(.*?)\"')

# Subject相关
pattern_cid = re.compile(r'\"cid\":(.*?),')
pattern_parent_subject = re.compile(r'parentSubject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_local_principal = re.compile(r'localPrincipal\":\"(.*?)\"')
pattern_local_principal_uuid = re.compile(r'localPrincipal\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_start_time = re.compile(r'startTimestampNanos\":(.*?),')
pattern_cmdline = re.compile(r'cmdLine\":{\"string\":\"(.*?)\"}')
pattern_path_prop = re.compile(r'\"path\":\"(.*?)\"')
pattern_ppid = re.compile(r'\"ppid\":\"(.*?)\"')
pattern_tgid = re.compile(r'\"tgid\":\"(.*?)\"')

# Principal相关
pattern_user_id = re.compile(r'userId\":\"(.*?)\"')
pattern_username = re.compile(r'username\":{\"string\":\"(.*?)\"}')
pattern_group_ids = re.compile(r'groupIds\":\[(.*?)\]')
pattern_cred = re.compile(r'\"cred\":\"(.*?)\"')

# FileObject相关
pattern_inode = re.compile(r'\"inode\":\"(.*?)\"')
pattern_dev = re.compile(r'\"dev\":\"(.*?)\"')
pattern_filename = re.compile(r'\"filename\":\"(.*?)\"')
pattern_permission = re.compile(r'permission\":{\"com.bbn.tc.schema.avro.cdm18.SHORT\":(.*?)}')
pattern_epoch = re.compile(r'epoch\":{\"int\":(.*?)}')

# MemoryObject相关
pattern_memory_address = re.compile(r'memoryAddress\":(.*?),')
pattern_mem_size = re.compile(r'size\":{\"long\":(.*?)}')

# NetFlowObject相关
pattern_local_address = re.compile(r'localAddress\":\"(.*?)\"')
pattern_local_port = re.compile(r'localPort\":(.*?),')
pattern_remote_address = re.compile(r'remoteAddress\":\"(.*?)\"')
pattern_remote_port = re.compile(r'remotePort\":(.*?),')
pattern_ip_protocol = re.compile(r'ipProtocol\":{\"int\":(.*?)}')


# ============================================================================
# 数据集元数据
# ============================================================================

metadata = {
    'theia': {
        'train': ['ta1-theia-e3-official-6r.json', 'ta1-theia-e3-official-6r.json.1',
                  'ta1-theia-e3-official-6r.json.2', 'ta1-theia-e3-official-6r.json.3',
                  'ta1-theia-e3-official-6r.json.6'],
        'test': ['ta1-theia-e3-official-6r.json.8','ta1-theia-e3-official-6r.json.12']
    },
    'cadets': {
        'train': ['ta1-cadets-e3-official.json', 'ta1-cadets-e3-official.json.1',
                  'ta1-cadets-e3-official.json.2', 'ta1-cadets-e3-official-2.json.1'],
        'test': ['ta1-cadets-e3-official-2.json']
    },
    'trace': {
        'train': ['ta1-trace-1-e5-official-1.json', 'ta1-trace-1-e5-official-1.json.1',
                  'ta1-trace-1-e5-official-1.json.2','ta1-trace-1-e5-official-1.json.3',
                  'ta1-trace-1-e5-official-1.json.6'],
        'test': ['ta1-trace-1-e5-official-1.json.4', 'ta1-trace-1-e5-official-1.json.5']
    }
}

# 事件类型分类（用于不变量计算）
READ_EVENTS = {'EVENT_READ', 'EVENT_RECVFROM', 'EVENT_RECVMSG', 'EVENT_READ_SOCKET_PARAMS'}
WRITE_EVENTS = {'EVENT_WRITE', 'EVENT_SENDTO', 'EVENT_SENDMSG', 'EVENT_WRITE_SOCKET_PARAMS'}
EXEC_EVENTS = {'EVENT_EXECUTE', 'EVENT_LOADLIBRARY'}
CREATE_EVENTS = {'EVENT_CREATE_OBJECT', 'EVENT_OPEN', 'EVENT_LSEEK'}
MMAP_EVENTS = {'EVENT_MMAP', 'EVENT_MPROTECT'}
CONNECT_EVENTS = {'EVENT_CONNECT', 'EVENT_ACCEPT'}
FORK_EVENTS = {'EVENT_FORK', 'EVENT_CLONE'}


# ============================================================================
# 全局状态
# ============================================================================

class ParserState:
    """解析器全局状态"""
    def __init__(self):
        # 类型映射
        self.node_type_dict: Dict[str, int] = {}
        self.edge_type_dict: Dict[str, int] = {}
        self.node_type_cnt = 0
        self.edge_type_cnt = 0
        
        # 实体存储
        self.principals: Dict[str, PrincipalInfo] = {}
        self.subjects: Dict[str, SubjectInfo] = {}
        self.file_objects: Dict[str, FileObjectInfo] = {}
        self.memory_objects: Dict[str, MemoryObjectInfo] = {}
        self.netflow_objects: Dict[str, NetFlowObjectInfo] = {}
        
        # 不变量追踪
        self.inode_to_files: Dict[str, List[str]] = defaultdict(list)  # inode -> [uuid, ...]
        # 访问历史用于 I_reach 等不变量；为保证结果正确性默认开启且不截断
        self.store_access_history: bool = True
        self.max_access_records_per_entity: int = -1  # 0=不保存，-1=不限制
        self.max_access_records_total: int = 0        # 0=不限制；达到后停止继续记录
        self._access_records_seen: int = 0

        self.file_access_history: Dict[str, Deque[AccessRecord]] = defaultdict(self._new_access_deque)  # file_uuid -> deque[AccessRecord]
        self.subject_access_history: Dict[str, Deque[AccessRecord]] = defaultdict(self._new_access_deque)  # subject_uuid -> deque[AccessRecord]
        
        # 名称映射
        self.id_nodetype_map: Dict[str, str] = {}
        self.id_nodename_map: Dict[str, str] = {}
        
        # 增强属性存储
        self.node_attributes: Dict[str, Dict] = {}  # uuid -> {attr_dict}
        self.edge_attributes: List[EdgeInfo] = []

    def _new_access_deque(self) -> Deque[AccessRecord]:
        if not self.store_access_history or self.max_access_records_per_entity == 0:
            return deque()
        if self.max_access_records_per_entity < 0:
            return deque()
        return deque(maxlen=self.max_access_records_per_entity)

    def can_record_access(self) -> bool:
        if not self.store_access_history or self.max_access_records_per_entity == 0:
            return False
        if self.max_access_records_total and self._access_records_seen >= self.max_access_records_total:
            return False
        return True


state = ParserState()

def _default_data_dir(dataset: str) -> str:
    """
    Resolve default dataset directory when project and datasets are separated.

    Priority:
    1) $DATA_ROOT/{dataset}
    2) /hy-tmp/data/{dataset} (common in some environments)
    3) ../data/{dataset} (legacy default)
    """
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        return os.path.join(os.path.expanduser(data_root), dataset)

    candidate = os.path.join(os.sep, "hy-tmp", "data", dataset)
    if os.path.exists(candidate):
        return candidate

    return os.path.join("..", "data", dataset)


# ============================================================================
# 工具函数
# ============================================================================

def safe_regex_find(pattern, text, default=""):
    """安全的正则匹配"""
    match = pattern.findall(text)
    return match[0] if match else default


def safe_int(value, default=0):
    """安全的整数转换"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def compute_hash(text: str) -> str:
    """计算字符串的MD5哈希"""
    if not text:
        return ""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def parse_cred(cred_str: str) -> Tuple[str, str, str, str, str, str]:
    """解析凭证字符串: euid/egid/suid/sgid/fsuid/fsgid/..."""
    parts = cred_str.split('/') if cred_str else []
    return (
        parts[0] if len(parts) > 0 else "",
        parts[1] if len(parts) > 1 else "",
        parts[2] if len(parts) > 2 else "",
        parts[3] if len(parts) > 3 else "",
        parts[4] if len(parts) > 4 else "",
        parts[5] if len(parts) > 5 else ""
    )


def classify_event(event_type: str) -> Dict[str, bool]:
    """事件类型分类"""
    return {
        'is_read': event_type in READ_EVENTS,
        'is_write': event_type in WRITE_EVENTS,
        'is_exec': event_type in EXEC_EVENTS,
        'is_create': event_type in CREATE_EVENTS,
        'is_mmap': event_type in MMAP_EVENTS,
        'is_connect': event_type in CONNECT_EVENTS,
        'is_send': event_type in {'EVENT_SENDTO', 'EVENT_SENDMSG'},
        'is_recv': event_type in {'EVENT_RECVFROM', 'EVENT_RECVMSG'}
    }


# ============================================================================
# 实体解析函数
# ============================================================================

def parse_principal(line: str) -> Optional[PrincipalInfo]:
    """解析Principal实体"""
    uuid = safe_regex_find(pattern_uuid, line)
    if not uuid or uuid == '00000000-0000-0000-0000-000000000000':
        return None
    
    user_id = safe_regex_find(pattern_user_id, line)
    username = safe_regex_find(pattern_username, line)
    cred = safe_regex_find(pattern_cred, line)
    host_id = safe_regex_find(pattern_host_id, line)
    
    # 解析group_ids
    group_ids_str = safe_regex_find(pattern_group_ids, line)
    group_ids = [g.strip().strip('"') for g in group_ids_str.split(',')] if group_ids_str else []
    
    # 解析凭证
    euid, egid, suid, sgid, fsuid, fsgid = parse_cred(cred)
    
    principal = PrincipalInfo(
        uuid=uuid,
        user_id=user_id,
        username=username,
        group_ids=group_ids,
        cred=cred,
        host_id=host_id,
        euid=euid,
        egid=egid,
        suid=suid,
        sgid=sgid,
        fsuid=fsuid,
        fsgid=fsgid
    )
    
    return principal


def parse_subject(line: str) -> Optional[SubjectInfo]:
    """解析Subject实体（进程）"""
    uuid = safe_regex_find(pattern_uuid, line)
    if not uuid or uuid == '00000000-0000-0000-0000-000000000000':
        return None
    
    subject_type = safe_regex_find(pattern_type, line)
    if subject_type == 'SUBJECT_UNIT':
        return None
    
    cid = safe_int(safe_regex_find(pattern_cid, line))
    parent_subject = safe_regex_find(pattern_parent_subject, line)
    host_id = safe_regex_find(pattern_host_id, line)
    
    # 尝试两种格式的localPrincipal
    local_principal = safe_regex_find(pattern_local_principal, line)
    if not local_principal:
        local_principal = safe_regex_find(pattern_local_principal_uuid, line)
    
    start_timestamp = safe_int(safe_regex_find(pattern_start_time, line))
    cmdline = safe_regex_find(pattern_cmdline, line)
    exe_path = safe_regex_find(pattern_path_prop, line)
    ppid = safe_regex_find(pattern_ppid, line)
    tgid = safe_regex_find(pattern_tgid, line)
    
    subject = SubjectInfo(
        uuid=uuid,
        subject_type=subject_type,
        cid=cid,
        ppid=ppid,
        tgid=tgid,
        parent_subject=parent_subject,
        host_id=host_id,
        local_principal=local_principal,
        start_timestamp=start_timestamp,
        cmdline=cmdline,
        exe_path=exe_path,
        exe_hash=compute_hash(exe_path)
    )
    
    return subject


def parse_file_object(line: str) -> Optional[FileObjectInfo]:
    """解析FileObject实体"""
    uuid = safe_regex_find(pattern_uuid, line)
    if not uuid or uuid == '00000000-0000-0000-0000-000000000000':
        return None
    
    file_type = safe_regex_find(pattern_type, line)
    if not file_type:
        if 'com.bbn.tc.schema.avro.cdm18.FileObject' in line:
            file_type = 'FILE_OBJECT_FILE'
    
    host_id = safe_regex_find(pattern_host_id, line)
    permission = safe_regex_find(pattern_permission, line)
    epoch = safe_int(safe_regex_find(pattern_epoch, line))
    
    inode = safe_regex_find(pattern_inode, line)
    dev = safe_regex_find(pattern_dev, line)
    filename = safe_regex_find(pattern_filename, line)
    
    # 尝试两种格式的localPrincipal
    local_principal = safe_regex_find(pattern_local_principal, line)
    if not local_principal:
        local_principal = safe_regex_find(pattern_local_principal_uuid, line)
    
    file_obj = FileObjectInfo(
        uuid=uuid,
        file_type=file_type,
        host_id=host_id,
        permission=permission,
        epoch=epoch,
        inode=inode,
        dev=dev,
        filename=filename,
        local_principal=local_principal
    )
    
    return file_obj


def parse_memory_object(line: str) -> Optional[MemoryObjectInfo]:
    """解析MemoryObject实体"""
    uuid = safe_regex_find(pattern_uuid, line)
    if not uuid or uuid == '00000000-0000-0000-0000-000000000000':
        return None
    
    host_id = safe_regex_find(pattern_host_id, line)
    memory_address = safe_int(safe_regex_find(pattern_memory_address, line))
    size = safe_int(safe_regex_find(pattern_mem_size, line))
    
    mem_obj = MemoryObjectInfo(
        uuid=uuid,
        host_id=host_id,
        memory_address=memory_address,
        size=size
    )
    
    return mem_obj


def parse_netflow_object(line: str) -> Optional[NetFlowObjectInfo]:
    """解析NetFlowObject实体"""
    uuid = safe_regex_find(pattern_uuid, line)
    if not uuid or uuid == '00000000-0000-0000-0000-000000000000':
        return None
    
    host_id = safe_regex_find(pattern_host_id, line)
    local_address = safe_regex_find(pattern_local_address, line)
    local_port = safe_int(safe_regex_find(pattern_local_port, line))
    remote_address = safe_regex_find(pattern_remote_address, line)
    remote_port = safe_int(safe_regex_find(pattern_remote_port, line))
    ip_protocol = safe_regex_find(pattern_ip_protocol, line)
    
    netflow_obj = NetFlowObjectInfo(
        uuid=uuid,
        host_id=host_id,
        local_address=local_address,
        local_port=local_port,
        remote_address=remote_address,
        remote_port=remote_port,
        ip_protocol=ip_protocol
    )
    
    return netflow_obj


def parse_event(line: str) -> Optional[EventInfo]:
    """解析Event实体"""
    uuid = safe_regex_find(pattern_uuid, line)
    if not uuid:
        return None
    
    event_type = safe_regex_find(pattern_type, line)
    sequence = safe_int(safe_regex_find(pattern_sequence, line))
    thread_id = safe_int(safe_regex_find(pattern_thread_id, line))
    host_id = safe_regex_find(pattern_host_id, line)
    timestamp = safe_int(safe_regex_find(pattern_time, line))
    
    subject_uuid = safe_regex_find(pattern_src, line)
    predicate_object_uuid = safe_regex_find(pattern_dst1, line)
    predicate_object2_uuid = safe_regex_find(pattern_dst2, line)
    predicate_object_path = safe_regex_find(pattern_dst1_path, line)
    predicate_object2_path = safe_regex_find(pattern_dst2_path, line)
    
    # 事件属性
    cmdline = safe_regex_find(pattern_cmdline_prop, line)
    flags = safe_regex_find(pattern_flags, line)
    mode = safe_regex_find(pattern_mode, line)
    prot = safe_regex_find(pattern_prot, line)
    return_code = safe_regex_find(pattern_rc, line)
    size = safe_int(safe_regex_find(pattern_size, line))
    
    # 事件分类
    event_class = classify_event(event_type)
    
    event = EventInfo(
        uuid=uuid,
        event_type=event_type,
        sequence=sequence,
        thread_id=thread_id,
        host_id=host_id,
        timestamp=timestamp,
        subject_uuid=subject_uuid,
        predicate_object_uuid=predicate_object_uuid,
        predicate_object2_uuid=predicate_object2_uuid,
        predicate_object_path=predicate_object_path,
        predicate_object2_path=predicate_object2_path,
        cmdline=cmdline,
        flags=flags,
        mode=mode,
        prot=prot,
        return_code=return_code,
        size=size,
        **event_class
    )
    
    return event


# ============================================================================
# 不变量追踪函数
# ============================================================================

def update_file_creator(file_uuid: str, subject_uuid: str, timestamp: int):
    """更新文件创建者信息"""
    if file_uuid not in state.file_objects:
        return
    
    file_obj = state.file_objects[file_uuid]
    if not file_obj.creator_uuid:  # 只记录第一个创建者
        file_obj.creator_uuid = subject_uuid
        file_obj.creator_timestamp = timestamp
        
        # 获取创建者的uid/gid/mnt_ns
        if subject_uuid in state.subjects:
            subject = state.subjects[subject_uuid]
            file_obj.creator_uid = subject.uid
            file_obj.creator_gid = subject.gid
            file_obj.creator_mnt_ns = subject.mnt_ns


def update_file_first_writer(file_uuid: str, subject_uuid: str, timestamp: int):
    """更新文件首写者信息"""
    if file_uuid not in state.file_objects:
        return
    
    file_obj = state.file_objects[file_uuid]
    if not file_obj.first_writer_uuid:  # 只记录第一个写者
        file_obj.first_writer_uuid = subject_uuid
        file_obj.first_write_timestamp = timestamp
        
        # 获取首写者的uid/gid/mnt_ns
        if subject_uuid in state.subjects:
            subject = state.subjects[subject_uuid]
            file_obj.first_writer_uid = subject.uid
            file_obj.first_writer_gid = subject.gid
            file_obj.first_writer_mnt_ns = subject.mnt_ns


def record_access(subject_uuid: str, object_uuid: str, access_type: str, timestamp: int):
    """记录访问记录（用于可达性不变量）"""
    if not state.can_record_access():
        return

    subject_uid = ""
    subject_mnt_ns = ""
    
    if subject_uuid in state.subjects:
        subject = state.subjects[subject_uuid]
        subject_uid = subject.uid
        subject_mnt_ns = subject.mnt_ns
    
    record = AccessRecord(
        subject_uuid=subject_uuid,
        object_uuid=object_uuid,
        access_type=access_type,
        timestamp=timestamp,
        subject_uid=subject_uid,
        subject_mnt_ns=subject_mnt_ns
    )
    
    state.file_access_history[object_uuid].append(record)
    state.subject_access_history[subject_uuid].append(record)
    state._access_records_seen += 1


def _iter_event_lines(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if "com.bbn.tc.schema.avro.cdm18.Event" in line:
                yield line


def update_inode_alias_map(file_obj: FileObjectInfo):
    """更新inode到文件的映射（用于多别名不变量）"""
    if file_obj.inode and file_obj.dev:
        key = f"{file_obj.dev}:{file_obj.inode}"
        if file_obj.uuid not in state.inode_to_files[key]:
            state.inode_to_files[key].append(file_obj.uuid)
        
        # 更新别名列表
        for uuid in state.inode_to_files[key]:
            if uuid != file_obj.uuid and uuid in state.file_objects:
                other_file = state.file_objects[uuid]
                if file_obj.filename and file_obj.filename not in other_file.aliases:
                    other_file.aliases.append(file_obj.filename)
                if other_file.filename and other_file.filename not in file_obj.aliases:
                    file_obj.aliases.append(other_file.filename)


def enrich_subject_with_principal(subject: SubjectInfo):
    """用Principal信息丰富Subject"""
    if subject.local_principal and subject.local_principal in state.principals:
        principal = state.principals[subject.local_principal]
        subject.uid = principal.user_id
        subject.gid = principal.group_ids[0] if principal.group_ids else ""
        subject.euid = principal.euid
        subject.egid = principal.egid


# ============================================================================
# 主解析流程
# ============================================================================

# 需要跳过的衍生文件（避免将输出文件当作原始日志再次解析）
GENERATED_FILES = {
    # JSON files
    'entities.json',
    'invariant_tracking.json',
    'names.json',
    'types.json',
    'metadata.json',
    # PKL files
    'train.pkl',
    'test.pkl',
    'malicious.pkl',
    'type_mappings.pkl',
    'names.pkl',
    'types.pkl',
    'entities.pkl',
    'invariant_tracking.pkl',
    'cic_scores.pkl',  # CIC不变量分数
}


# ============================================================================
# 元数据/缓存文件（PKL）辅助
# ============================================================================

def _artifact_candidate_paths(data_dir: str, filename: str) -> List[str]:
    """
    兼容不同的保存布局：
    - {data_dir}/{filename}
    - {data_dir}/{stem}/{filename}              (例如 names/names.pkl)
    - {data_dir}/meta/{filename} 或 metadata/pkl/cache/{filename}
    """
    stem = filename[:-4] if filename.lower().endswith(".pkl") else filename
    return [
        os.path.join(data_dir, filename),
        os.path.join(data_dir, stem, filename),
        os.path.join(data_dir, "meta", filename),
        os.path.join(data_dir, "metadata", filename),
        os.path.join(data_dir, "pkl", filename),
        os.path.join(data_dir, "cache", filename),
    ]


def _resolve_existing_artifact_path(data_dir: str, filename: str) -> Optional[str]:
    for path in _artifact_candidate_paths(data_dir, filename):
        if os.path.exists(path):
            return path
    return None


def _default_artifact_save_path(data_dir: str, filename: str) -> str:
    """
    若用户已经把文件移动到子目录（例如 names/names.pkl），则继续写回该位置；
    否则默认写到 {data_dir}/{filename}。
    """
    existing = _resolve_existing_artifact_path(data_dir, filename)
    return existing or os.path.join(data_dir, filename)


def _load_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        return pkl.load(f)


def _save_pkl(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pkl.dump(obj, f)


def _load_entities_into_state(entities_data: Dict[str, Any]):
    # entities.pkl 由 asdict() 序列化而来，这里重建为 dataclass 对象
    principals = entities_data.get("principals", {}) or {}
    subjects = entities_data.get("subjects", {}) or {}
    file_objects = entities_data.get("file_objects", {}) or {}
    memory_objects = entities_data.get("memory_objects", {}) or {}
    netflow_objects = entities_data.get("netflow_objects", {}) or {}

    state.principals = {k: PrincipalInfo(**v) for k, v in principals.items()}
    state.subjects = {k: SubjectInfo(**v) for k, v in subjects.items()}
    state.file_objects = {k: FileObjectInfo(**v) for k, v in file_objects.items()}
    state.memory_objects = {k: MemoryObjectInfo(**v) for k, v in memory_objects.items()}
    state.netflow_objects = {k: NetFlowObjectInfo(**v) for k, v in netflow_objects.items()}

    # 若 types.pkl / names.pkl 缺失，可从实体信息重建基础映射
    if not state.id_nodetype_map:
        for uuid, subject in state.subjects.items():
            state.id_nodetype_map[uuid] = subject.subject_type
        for uuid, file_obj in state.file_objects.items():
            state.id_nodetype_map[uuid] = file_obj.file_type
        for uuid in state.memory_objects:
            state.id_nodetype_map[uuid] = "MemoryObject"
        for uuid in state.netflow_objects:
            state.id_nodetype_map[uuid] = "NetFlowObject"

    if not state.id_nodename_map:
        for uuid, subject in state.subjects.items():
            if subject.exe_path:
                state.id_nodename_map[uuid] = subject.exe_path
        for uuid, file_obj in state.file_objects.items():
            if file_obj.filename:
                state.id_nodename_map[uuid] = file_obj.filename
        for uuid, netflow in state.netflow_objects.items():
            if netflow.remote_address:
                state.id_nodename_map[uuid] = netflow.remote_address

    # inode->files + aliases（若缓存里没有，也可以重建）
    for file_obj in state.file_objects.values():
        update_inode_alias_map(file_obj)

    for subject in state.subjects.values():
        enrich_subject_with_principal(subject)


def load_enhanced_metadata_cache(data_dir: str) -> bool:
    """
    加载缓存的 entities/names/types/invariant_tracking（若存在）。
    返回 True 表示已加载 entities.pkl，可跳过 Phase 1 实体扫描。
    """
    entities_path = _resolve_existing_artifact_path(data_dir, "entities.pkl")
    if not entities_path:
        return False

    # 可选：names/types/invariant_tracking
    names_path = _resolve_existing_artifact_path(data_dir, "names.pkl")
    types_path = _resolve_existing_artifact_path(data_dir, "types.pkl")
    invariant_path = _resolve_existing_artifact_path(data_dir, "invariant_tracking.pkl")

    if names_path:
        names_data = _load_pkl(names_path) or {}
        state.id_nodename_map = names_data.get("id_nodename_map", {}) or {}

    if types_path:
        types_data = _load_pkl(types_path) or {}
        state.id_nodetype_map = types_data.get("id_nodetype_map", {}) or {}

    entities_data = _load_pkl(entities_path) or {}
    _load_entities_into_state(entities_data)

    if invariant_path:
        invariant_data = _load_pkl(invariant_path) or {}
        inode_to_files = invariant_data.get("inode_to_files", {}) or {}
        state.inode_to_files = defaultdict(list, inode_to_files)

        file_hist = invariant_data.get("file_access_history", {}) or {}
        subject_hist = invariant_data.get("subject_access_history", {}) or {}
        state.file_access_history = defaultdict(state._new_access_deque)
        state.subject_access_history = defaultdict(state._new_access_deque)
        for k, v in file_hist.items():
            dq = state._new_access_deque()
            dq.extend(AccessRecord(**r) for r in v)
            state.file_access_history[k] = dq
        for k, v in subject_hist.items():
            dq = state._new_access_deque()
            dq.extend(AccessRecord(**r) for r in v)
            state.subject_access_history[k] = dq

    print(f"[Cache] 已加载实体元数据: {os.path.relpath(entities_path, data_dir)}")
    if names_path:
        print(f"[Cache] 已加载 names: {os.path.relpath(names_path, data_dir)}")
    if types_path:
        print(f"[Cache] 已加载 types: {os.path.relpath(types_path, data_dir)}")
    if invariant_path:
        print(f"[Cache] 已加载 invariant_tracking: {os.path.relpath(invariant_path, data_dir)}")

    return True


def preprocess_dataset_enhanced(
    dataset: str,
    data_dir: str,
    use_meta_cache: bool = True,
    force_rebuild_meta: bool = False,
):
    """增强版数据集预处理（支持从PKL缓存加载实体/不变量元数据）"""
    loaded_entities_cache = False
    loaded_invariant_cache = False

    if use_meta_cache and not force_rebuild_meta:
        loaded_entities_cache = load_enhanced_metadata_cache(data_dir)
        if loaded_entities_cache:
            loaded_invariant_cache = _resolve_existing_artifact_path(data_dir, "invariant_tracking.pkl") is not None

    if loaded_entities_cache:
        print("[Phase 1] 使用缓存实体元数据，跳过实体扫描。")
    else:
        print("[Phase 1] 扫描实体信息...")

        # 第一遍：收集所有实体（含额外分片 .json.N，用于补全实体定义）
        for file in os.listdir(data_dir):
            # 跳过非JSON文件和衍生文件
            if not file.endswith(".json") and ".json." not in file:
                continue
            if file in GENERATED_FILES:
                print(f"  跳过衍生文件: {file}")
                continue
            if file.endswith(".txt") or "names" in file or "types" in file or "metadata" in file:
                continue
            # 跳过压缩文件
            if file.endswith(".tar.gz") or file.endswith(".gz"):
                continue

            filepath = os.path.join(data_dir, file)
            print(f"  读取实体: {file}")

            with open(filepath, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"  {file}"):
                    # 跳过非实体行
                    if "com.bbn.tc.schema.avro.cdm18.Event" in line:
                        continue
                    if "com.bbn.tc.schema.avro.cdm18.Host" in line:
                        continue
                    if "com.bbn.tc.schema.avro.cdm18.TimeMarker" in line:
                        continue
                    if "com.bbn.tc.schema.avro.cdm18.StartMarker" in line:
                        continue
                    if "com.bbn.tc.schema.avro.cdm18.EndMarker" in line:
                        continue
                    if "com.bbn.tc.schema.avro.cdm18.UnitDependency" in line:
                        continue

                    # 解析Principal
                    if "com.bbn.tc.schema.avro.cdm18.Principal" in line:
                        principal = parse_principal(line)
                        if principal:
                            state.principals[principal.uuid] = principal
                        continue

                    # 解析Subject
                    if "com.bbn.tc.schema.avro.cdm18.Subject" in line:
                        subject = parse_subject(line)
                        if subject:
                            state.subjects[subject.uuid] = subject
                            state.id_nodetype_map[subject.uuid] = subject.subject_type
                            if subject.exe_path:
                                state.id_nodename_map[subject.uuid] = subject.exe_path
                        continue

                    # 解析FileObject
                    if "com.bbn.tc.schema.avro.cdm18.FileObject" in line:
                        file_obj = parse_file_object(line)
                        if file_obj:
                            state.file_objects[file_obj.uuid] = file_obj
                            state.id_nodetype_map[file_obj.uuid] = file_obj.file_type
                            if file_obj.filename:
                                state.id_nodename_map[file_obj.uuid] = file_obj.filename
                            update_inode_alias_map(file_obj)
                        continue

                    # 解析MemoryObject
                    if "com.bbn.tc.schema.avro.cdm18.MemoryObject" in line:
                        mem_obj = parse_memory_object(line)
                        if mem_obj:
                            state.memory_objects[mem_obj.uuid] = mem_obj
                            state.id_nodetype_map[mem_obj.uuid] = "MemoryObject"
                        continue

                    # 解析NetFlowObject
                    if "com.bbn.tc.schema.avro.cdm18.NetFlowObject" in line:
                        netflow_obj = parse_netflow_object(line)
                        if netflow_obj:
                            state.netflow_objects[netflow_obj.uuid] = netflow_obj
                            state.id_nodetype_map[netflow_obj.uuid] = "NetFlowObject"
                            if netflow_obj.remote_address:
                                state.id_nodename_map[netflow_obj.uuid] = netflow_obj.remote_address
                        continue

                    # 解析UnnamedPipeObject（仅记录类型，实体本身不落盘）
                    if "com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject" in line:
                        uuid = safe_regex_find(pattern_uuid, line)
                        if uuid and uuid != "00000000-0000-0000-0000-000000000000":
                            state.id_nodetype_map[uuid] = "UnnamedPipeObject"

        # 用Principal信息丰富Subject
        print("[Phase 1.5] 丰富Subject信息...")
        for subject in state.subjects.values():
            enrich_subject_with_principal(subject)

    # 第二遍：扫描Event仅更新不变量，不再把所有边存入内存（避免OOM）
    if loaded_invariant_cache:
        print("[Phase 2] 使用缓存不变量，跳过事件扫描。")
        return

    print("[Phase 2] 扫描事件并更新不变量...")
    all_files: List[str] = []
    for split in metadata.get(dataset, {}):
        all_files.extend(metadata[dataset][split])

    for file in all_files:
        filepath = os.path.join(data_dir, file)
        if not os.path.exists(filepath):
            continue
        print(f"  处理事件: {file}")
        for line in tqdm(_iter_event_lines(filepath), desc=f"  {file}"):
            event = parse_event(line)
            if not event or not event.subject_uuid:
                continue
            src_uuid = event.subject_uuid
            if src_uuid not in state.id_nodetype_map:
                continue
            if event.predicate_object_uuid and event.predicate_object_uuid in state.id_nodetype_map:
                update_invariant_tracking(event, src_uuid, event.predicate_object_uuid)
            if event.predicate_object2_uuid and event.predicate_object2_uuid in state.id_nodetype_map:
                update_invariant_tracking(event, src_uuid, event.predicate_object2_uuid)


def build_enhanced_edge(event: EventInfo, src_uuid: str, src_type: str, 
                        dst_uuid: str, dst_type: str, reversed_edge: bool = False) -> EdgeInfo:
    """
    构建增强边信息
    
    Args:
        event: 事件信息
        src_uuid: 最终的源UUID（可能已反转）
        src_type: 最终的源类型
        dst_uuid: 最终的目标UUID（可能已反转）
        dst_type: 最终的目标类型
        reversed_edge: 边是否已反转（用于正确获取属性）
    """
    edge = EdgeInfo(
        src_uuid=src_uuid,
        dst_uuid=dst_uuid,
        src_type=src_type,
        dst_type=dst_type,
        edge_type=event.event_type,
        timestamp=event.timestamp,
        flags=event.flags,
        mode=event.mode,
        prot=event.prot,
        size=event.size,
        return_code=event.return_code
    )
    
    # 根据是否反转，确定原始的subject和object UUID
    if reversed_edge:
        # 反转后：src_uuid是原来的object，dst_uuid是原来的subject
        original_subject_uuid = dst_uuid
        original_object_uuid = src_uuid
    else:
        # 未反转：src_uuid是subject，dst_uuid是object
        original_subject_uuid = src_uuid
        original_object_uuid = dst_uuid
    
    # 源节点属性（最终图中的src）
    if src_uuid in state.subjects:
        subject = state.subjects[src_uuid]
        edge.src_uid = subject.uid
        edge.src_gid = subject.gid
        edge.src_mnt_ns = subject.mnt_ns
        edge.src_exe_hash = subject.exe_hash
    elif src_uuid in state.file_objects:
        # 反转边时，src可能是文件
        file_obj = state.file_objects[src_uuid]
        edge.src_uid = file_obj.creator_uid
        edge.src_gid = file_obj.creator_gid
        edge.src_mnt_ns = file_obj.creator_mnt_ns
        edge.src_exe_hash = ""
    
    # 目标节点属性（最终图中的dst）
    if dst_uuid in state.file_objects:
        file_obj = state.file_objects[dst_uuid]
        edge.dst_inode = file_obj.inode
        edge.dst_path = file_obj.filename
    elif dst_uuid in state.subjects:
        # 反转边时，dst可能是进程
        subject = state.subjects[dst_uuid]
        edge.dst_inode = ""
        edge.dst_path = subject.exe_path
    elif dst_uuid in state.netflow_objects:
        netflow = state.netflow_objects[dst_uuid]
        edge.dst_inode = ""
        edge.dst_path = f"{netflow.remote_address}:{netflow.remote_port}"
    
    return edge


def update_invariant_tracking(event: EventInfo, src_uuid: str, dst_uuid: str):
    """更新不变量追踪信息"""
    # 记录访问（用于可达性不变量）
    if event.is_read:
        record_access(src_uuid, dst_uuid, 'read', event.timestamp)
    if event.is_write:
        record_access(src_uuid, dst_uuid, 'write', event.timestamp)
        update_file_first_writer(dst_uuid, src_uuid, event.timestamp)
    if event.is_exec:
        record_access(src_uuid, dst_uuid, 'exec', event.timestamp)
    if event.is_mmap:
        record_access(src_uuid, dst_uuid, 'mmap', event.timestamp)
    if event.is_create:
        record_access(src_uuid, dst_uuid, 'create', event.timestamp)
        update_file_creator(dst_uuid, src_uuid, event.timestamp)


def save_enhanced_metadata(dataset: str, data_dir: str):
    """保存增强的元数据为PKL文件"""
    print("[Phase 3] 保存元数据...")
    
    # 打印统计信息
    print(f"  - Principal数量: {len(state.principals)}")
    print(f"  - Subject数量: {len(state.subjects)}")
    print(f"  - FileObject数量: {len(state.file_objects)}")
    print(f"  - MemoryObject数量: {len(state.memory_objects)}")
    print(f"  - NetFlowObject数量: {len(state.netflow_objects)}")
    print(f"  - inode别名组: {len(state.inode_to_files)}")
    print(f"  - 文件访问记录: {sum(len(v) for v in state.file_access_history.values())}")
    
    # 保存names.pkl - 节点名称映射
    names_data = {
        'id_nodename_map': state.id_nodename_map,
    }
    names_path = _default_artifact_save_path(data_dir, "names.pkl")
    _save_pkl(names_path, names_data)
    print(f"  已保存 {os.path.relpath(names_path, data_dir)}")
    
    # 保存types.pkl - UUID->类型映射（注意：最终图的数值type映射见 type_mappings.pkl）
    types_data = {
        'id_nodetype_map': state.id_nodetype_map,
    }
    types_path = _default_artifact_save_path(data_dir, "types.pkl")
    _save_pkl(types_path, types_data)
    print(f"  已保存 {os.path.relpath(types_path, data_dir)}")
    
    # 保存entities.pkl - 实体信息（使用dataclass的asdict转换）
    entities_data = {
        'principals': {k: asdict(v) for k, v in state.principals.items()},
        'subjects': {k: asdict(v) for k, v in state.subjects.items()},
        'file_objects': {k: asdict(v) for k, v in state.file_objects.items()},
        'memory_objects': {k: asdict(v) for k, v in state.memory_objects.items()},
        'netflow_objects': {k: asdict(v) for k, v in state.netflow_objects.items()},
    }
    entities_path = _default_artifact_save_path(data_dir, "entities.pkl")
    _save_pkl(entities_path, entities_data)
    print(f"  已保存 {os.path.relpath(entities_path, data_dir)}")
    
    # 保存invariant_tracking.pkl - 不变量追踪信息
    invariant_data = {
        'inode_to_files': dict(state.inode_to_files),
        'file_access_history': {k: [asdict(r) for r in list(v)] for k, v in state.file_access_history.items()},
        'subject_access_history': {k: [asdict(r) for r in list(v)] for k, v in state.subject_access_history.items()},
    }
    invariant_path = _default_artifact_save_path(data_dir, "invariant_tracking.pkl")
    _save_pkl(invariant_path, invariant_data)
    print(f"  已保存 {os.path.relpath(invariant_path, data_dir)}")


def _build_node_attrs(uuid: str, type_name: str, type_id: int) -> Dict[str, Any]:
    node_attrs: Dict[str, Any] = {'type': type_id, 'uuid': uuid, 'type_name': type_name}
    if uuid in state.subjects:
        subj = state.subjects[uuid]
        node_attrs.update({
            'uid': subj.uid,
            'gid': subj.gid,
            'mnt_ns': subj.mnt_ns,
            'pid_ns': subj.pid_ns,
            'exe_hash': subj.exe_hash,
            'cmdline': subj.cmdline,
            'exe_path': subj.exe_path,
            'ppid': subj.ppid,
            'start_timestamp': subj.start_timestamp,
        })
    elif uuid in state.file_objects:
        fobj = state.file_objects[uuid]
        node_attrs.update({
            'inode': fobj.inode,
            'dev': fobj.dev,
            'filename': fobj.filename,
            'permission': fobj.permission,
            'creator_uid': fobj.creator_uid,
            'creator_gid': fobj.creator_gid,
            'creator_mnt_ns': fobj.creator_mnt_ns,
            'creator_timestamp': fobj.creator_timestamp,
            'first_writer_uid': fobj.first_writer_uid,
            'first_writer_gid': fobj.first_writer_gid,
            'first_writer_mnt_ns': fobj.first_writer_mnt_ns,
            'first_write_timestamp': fobj.first_write_timestamp,
            'alias_count': len(fobj.aliases),
        })
    elif uuid in state.netflow_objects:
        nobj = state.netflow_objects[uuid]
        node_attrs.update({
            'local_address': nobj.local_address,
            'local_port': nobj.local_port,
            'remote_address': nobj.remote_address,
            'remote_port': nobj.remote_port,
        })
    return node_attrs


def read_single_graph_enhanced(
    dataset: str,
    malicious: Set[str],
    path: str,
    data_dir: str,
    test: bool = False,
    update_invariants: bool = False,
):
    """
    增强版单图读取（流式从原始日志构图，避免边/排序列表导致的内存膨胀）

    - 使用MultiDiGraph支持多重边，保留所有事件
    - 读取/接收/加载类事件反转方向（保持“信息流”一致）
    """
    g = nx.MultiDiGraph()
    print(f'  转换图: {path}')

    filepath = os.path.join(data_dir, path)
    if not os.path.exists(filepath):
        print(f'    警告: 未找到文件: {path}')
        return {}, g

    node_map: Dict[str, int] = {}
    node_cnt = 0

    def ensure_node(uuid: str, type_name: str) -> int:
        nonlocal node_cnt
        if type_name not in state.node_type_dict:
            state.node_type_dict[type_name] = state.node_type_cnt
            state.node_type_cnt += 1
        if uuid not in node_map:
            type_id = state.node_type_dict[type_name]
            node_map[uuid] = node_cnt
            g.add_node(node_cnt, **_build_node_attrs(uuid, type_name, type_id))
            node_cnt += 1
        return node_map[uuid]

    for line in tqdm(_iter_event_lines(filepath), desc=f"  {path}"):
        event = parse_event(line)
        if not event or not event.subject_uuid:
            continue

        src_uuid = event.subject_uuid
        if src_uuid not in state.id_nodetype_map:
            continue
        src_type = state.id_nodetype_map[src_uuid]

        should_reverse = (
            "READ" in event.event_type or "RECV" in event.event_type or "LOAD" in event.event_type
        )

        for dst_uuid in (event.predicate_object_uuid, event.predicate_object2_uuid):
            if not dst_uuid or dst_uuid not in state.id_nodetype_map:
                continue
            dst_type = state.id_nodetype_map[dst_uuid]

            if update_invariants:
                update_invariant_tracking(event, src_uuid, dst_uuid)

            if not test:
                if src_uuid in malicious and src_type != 'MemoryObject':
                    continue
                if dst_uuid in malicious and dst_type != 'MemoryObject':
                    continue

            if should_reverse:
                final_src, final_dst = dst_uuid, src_uuid
                final_src_type, final_dst_type = dst_type, src_type
            else:
                final_src, final_dst = src_uuid, dst_uuid
                final_src_type, final_dst_type = src_type, dst_type

            if event.event_type not in state.edge_type_dict:
                state.edge_type_dict[event.event_type] = state.edge_type_cnt
                state.edge_type_cnt += 1
            edge_type_id = state.edge_type_dict[event.event_type]

            edge_info = build_enhanced_edge(
                event, final_src, final_src_type, final_dst, final_dst_type, should_reverse
            )

            src_id = ensure_node(final_src, final_src_type)
            dst_id = ensure_node(final_dst, final_dst_type)

            edge_attrs = {
                'type': edge_type_id,
                'timestamp': event.timestamp,
                'edge_type_name': event.event_type,
                'flags': edge_info.flags,
                'mode': edge_info.mode,
                'prot': edge_info.prot,
                'size': edge_info.size,
                'return_code': edge_info.return_code,
                'src_uid': edge_info.src_uid,
                'src_gid': edge_info.src_gid,
                'src_mnt_ns': edge_info.src_mnt_ns,
                'src_exe_hash': edge_info.src_exe_hash,
                'dst_inode': edge_info.dst_inode,
                'dst_path': edge_info.dst_path,
            }

            g.add_edge(src_id, dst_id, **edge_attrs)

    return node_map, g


def read_graphs_enhanced(
    dataset: str,
    data_dir: str,
    use_meta_cache: bool = True,
    force_rebuild_meta: bool = False,
):
    """增强版图读取主函数"""
    # 读取恶意实体列表
    malicious_file = os.path.join(data_dir, f'{dataset}.txt')
    malicious_entities = set()
    if os.path.exists(malicious_file):
        with open(malicious_file, 'r') as f:
            for l in f:
                malicious_entities.add(l.strip())
    
    # 预处理数据集
    preprocess_dataset_enhanced(
        dataset,
        data_dir,
        use_meta_cache=use_meta_cache,
        force_rebuild_meta=force_rebuild_meta,
    )
    
    # 保存增强元数据
    save_enhanced_metadata(dataset, data_dir)
    
    # 读取训练图
    print("\n[Phase 4] 构建训练图...")
    train_gs = []
    for file in metadata[dataset]['train']:
        if os.path.exists(os.path.join(data_dir, file)):
            _, train_g = read_single_graph_enhanced(dataset, malicious_entities, file, data_dir, False)
            train_gs.append(train_g)

    # 读取测试图
    print("\n[Phase 5] 构建测试图...")
    test_gs = []
    test_node_map = {}
    count_node = 0
    for file in metadata[dataset]['test']:
        if os.path.exists(os.path.join(data_dir, file)):
            node_map, test_g = read_single_graph_enhanced(dataset, malicious_entities, file, data_dir, True)
            test_gs.append(test_g)
            for key in node_map:
                if key not in test_node_map:
                    test_node_map[key] = node_map[key] + count_node
            count_node += test_g.number_of_nodes()
    
    # 处理恶意实体
    print("\n[Phase 6] 处理恶意实体...")
    final_malicious_entities = []
    malicious_names = []
    
    for e in malicious_entities:
        if e in test_node_map:
            if e in state.id_nodetype_map:
                if state.id_nodetype_map[e] not in ['MemoryObject', 'UnnamedPipeObject']:
                    final_malicious_entities.append(test_node_map[e])
                    name = state.id_nodename_map.get(e, e)
                    malicious_names.append(name)
    
    # 保存结果（MultiDiGraph需要特殊处理以保留多重边的key）
    print("\n[Phase 7] 保存最终结果...")
    _save_pkl(os.path.join(data_dir, "malicious.pkl"), (final_malicious_entities, malicious_names))
    # 对于MultiDiGraph，node_link_data会自动处理多重边
    _save_pkl(os.path.join(data_dir, "train.pkl"), [nx.node_link_data(g) for g in train_gs])
    _save_pkl(os.path.join(data_dir, "test.pkl"), [nx.node_link_data(g) for g in test_gs])
    
    # 保存类型映射和元信息
    _save_pkl(os.path.join(data_dir, "type_mappings.pkl"), {
        'node_type_dict': state.node_type_dict,
        'edge_type_dict': state.edge_type_dict,
        'graph_type': 'MultiDiGraph',  # 标记图类型
        'supports_multi_edges': True,   # 支持多重边
    })
    
    print("\n=== 处理完成 ===")
    print(f"训练图数量: {len(train_gs)}")
    print(f"测试图数量: {len(test_gs)}")
    if train_gs:
        total_train_nodes = sum(g.number_of_nodes() for g in train_gs)
        total_train_edges = sum(g.number_of_edges() for g in train_gs)
        print(f"训练图总节点数: {total_train_nodes}")
        print(f"训练图总边数: {total_train_edges}")
    if test_gs:
        total_test_nodes = sum(g.number_of_nodes() for g in test_gs)
        total_test_edges = sum(g.number_of_edges() for g in test_gs)
        print(f"测试图总节点数: {total_test_nodes}")
        print(f"测试图总边数: {total_test_edges}")
    print(f"恶意实体数量: {len(final_malicious_entities)}")
    print(f"节点类型数量: {len(state.node_type_dict)}")
    print(f"边类型数量: {len(state.edge_type_dict)}")
    print(f"图类型: MultiDiGraph (支持多重边)")




# ============================================================================
# 工具函数：加载图数据
# ============================================================================

def load_graphs(data_dir: str, split: str = 'train') -> List[nx.MultiDiGraph]:
    """
    加载保存的图数据
    
    Args:
        data_dir: 数据目录
        split: 'train' 或 'test'
        
    Returns:
        MultiDiGraph列表
    """
    filepath = os.path.join(data_dir, f'{split}.pkl')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到文件: {filepath}")
    
    with open(filepath, 'rb') as f:
        graph_data_list = pkl.load(f)
    
    # 将node_link_data转换回MultiDiGraph
    graphs = []
    for g_data in graph_data_list:
        # 检查是否为多重图数据
        if g_data.get('multigraph', False):
            g = nx.node_link_graph(g_data, multigraph=True, directed=True)
        else:
            g = nx.node_link_graph(g_data)
        graphs.append(g)
    
    return graphs


def get_edge_events(graph: nx.MultiDiGraph, src: int, dst: int) -> List[Dict]:
    """
    获取两个节点之间的所有事件
    
    Args:
        graph: MultiDiGraph
        src: 源节点ID
        dst: 目标节点ID
        
    Returns:
        事件列表，按时间戳排序
    """
    if not graph.has_edge(src, dst):
        return []
    
    # MultiDiGraph中，边数据是一个字典，key是边的索引
    edge_data = graph.get_edge_data(src, dst)
    events = list(edge_data.values())
    
    # 按时间戳排序
    events.sort(key=lambda e: e.get('timestamp', 0))
    
    return events


def aggregate_edges(graph: nx.MultiDiGraph, 
                    method: str = 'latest') -> nx.DiGraph:
    """
    将MultiDiGraph聚合为DiGraph
    
    Args:
        graph: MultiDiGraph
        method: 聚合方法
            - 'latest': 保留最新的边
            - 'earliest': 保留最早的边
            - 'count': 保留边计数
            
    Returns:
        聚合后的DiGraph
    """
    simple_graph = nx.DiGraph()
    
    # 复制节点
    for node, data in graph.nodes(data=True):
        simple_graph.add_node(node, **data)
    
    # 聚合边
    for src, dst in set(graph.edges()):
        events = get_edge_events(graph, src, dst)
        if not events:
            continue
        
        if method == 'latest':
            edge_data = max(events, key=lambda e: e.get('timestamp', 0))
        elif method == 'earliest':
            edge_data = min(events, key=lambda e: e.get('timestamp', 0))
        elif method == 'count':
            edge_data = events[0].copy()
            edge_data['event_count'] = len(events)
            edge_data['timestamps'] = [e.get('timestamp', 0) for e in events]
        else:
            edge_data = events[0]
        
        simple_graph.add_edge(src, dst, **edge_data)
    
    return simple_graph


# ============================================================================
# 入口点
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced CDM Parser for CIC Analysis')
    parser.add_argument("--dataset", type=str, default="theia",
                        choices=['trace', 'theia', 'cadets', 'clear'],
                        help="Dataset name")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory path (default: ../data/{dataset}/）")
    parser.add_argument(
        "--no_meta_cache",
        action="store_true",
        help="Disable loading entities/names/types/invariant_tracking from *.pkl cache",
    )
    parser.add_argument(
        "--force_rebuild_meta",
        action="store_true",
        help="Ignore existing meta cache and rebuild from raw logs",
    )
    args = parser.parse_args()
    
    # 如果未指定data_dir，则根据dataset自动设置
    if args.data_dir is None:
        args.data_dir = _default_data_dir(args.dataset)
    
    print(f"使用数据集: {args.dataset}")
    print(f"数据目录: {args.data_dir}")
    
    # 重置全局状态
    state = ParserState()

    read_graphs_enhanced(
        args.dataset,
        args.data_dir,
        use_meta_cache=not args.no_meta_cache,
        force_rebuild_meta=args.force_rebuild_meta,
    )
