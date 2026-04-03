"""
Connected Minimal Explanation Subgraph Builder

Research design implementation:
1. Initialize: Select Top-K anomaly nodes as seeds S₀
2. Expand: Greedy edge selection (ΔP/ΔC) to connect seeds
3. Evidence completion: Add evidence nodes for anomaly factors
4. Post-pruning: Remove redundant leaves

Three-layer visualization:
- Seed layer (red border)
- Expansion layer (black dashed)
- Final layer (black solid) + Pruned layer (gray dashed)
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any

import torch
import networkx as nx
import dgl


@dataclass
class ConnectedSubgraphLayers:
    """Three-layer structure for visualization"""
    seeds: Set[int]  # S₀ seed nodes (red)
    expanded_nodes: Set[int]  # Expanded nodes
    expanded_edges: Set[Tuple[int, int]]  # Expanded edges
    final_nodes: Set[int]  # Final nodes (solid)
    final_edges: Set[Tuple[int, int]]  # Final edges (solid)
    pruned_nodes: Set[int]  # Pruned nodes (gray)
    pruned_edges: Set[Tuple[int, int]]  # Pruned edges (dashed)
    attack_path: List[int] = field(default_factory=list)


# Node type mapping
NODE_TYPE_NAMES = {
    0: 'subject',
    1: 'file', 
    2: 'netflow',
    3: 'memory',
    4: 'principal',
}


class ConnectedExplanationBuilder:
    """
    Connected Minimal Explanation Subgraph Builder
    
    Research design:
    1. Initialize: Select Top-K anomaly nodes as seeds S₀
    2. Expand: Greedy edge selection to connect seeds
    3. Evidence completion: Add evidence nodes
    4. Post-pruning: Remove low-value leaves
    """
    
    def __init__(
        self,
        graph: dgl.DGLGraph,
        anomaly_scores: torch.Tensor,
        cic_scores: Optional[torch.Tensor] = None,
        names_map: Optional[Dict[int, str]] = None,
        types_map: Optional[Dict[int, int]] = None,
        alpha: float = 0.5,
        bridge_budget: int = 15,
    ):
        self.graph = graph
        self.anomaly_scores = torch.nan_to_num(anomaly_scores, nan=0.0).clamp(0.0, 1.0)
        self.cic_scores = cic_scores
        self.names_map = names_map or {}
        self.types_map = types_map or {}
        self.alpha = alpha
        self.bridge_budget = bridge_budget
        
        self._node_rewards = self._compute_node_rewards()
        self._build_adjacency()
    
    def _compute_node_rewards(self) -> torch.Tensor:
        """Compute node rewards: p_x = α·CIC(x) + (1-α)·anom(x)"""
        anom = self.anomaly_scores.float()
        if self.cic_scores is not None and self.cic_scores.dim() == 2:
            cic = 1.0 - (1.0 - 0.25 * self.cic_scores.clamp(0, 1)).prod(dim=1)
            cic = cic.clamp(0.0, 1.0)
        else:
            cic = torch.zeros_like(anom)
        return (self.alpha * cic + (1 - self.alpha) * anom).clamp(0.0, 1.0)
    
    def _build_adjacency(self):
        """Build adjacency lists"""
        src, dst = self.graph.edges()
        self._src = src.tolist()
        self._dst = dst.tolist()
        
        self._adj_out: Dict[int, List[Tuple[int, int]]] = {}
        self._adj_in: Dict[int, List[Tuple[int, int]]] = {}
        
        for eidx, (s, d) in enumerate(zip(self._src, self._dst)):
            self._adj_out.setdefault(s, []).append((d, eidx))
            self._adj_in.setdefault(d, []).append((s, eidx))
    
    def _get_neighbors(self, node: int) -> List[Tuple[int, int]]:
        neighbors = []
        neighbors.extend(self._adj_out.get(node, []))
        neighbors.extend(self._adj_in.get(node, []))
        return neighbors
    
    def _compute_edge_reward(self, src: int, dst: int) -> float:
        return float(self._node_rewards[src].item() + self._node_rewards[dst].item()) / 2.0
    
    def build(
        self,
        top_k: int = 5,
        anomaly_threshold: float = 0.5,
    ) -> Tuple[Any, ConnectedSubgraphLayers]:
        """Build connected minimal explanation subgraph"""
        
        # Step 1: Initialize seeds S₀
        scores = self.anomaly_scores.cpu()
        top_k = min(top_k, scores.size(0))
        _, seed_indices = torch.topk(scores, top_k)
        seeds = set(int(idx.item()) for idx in seed_indices)
        
        print(f"[Connected] Seeds S0: {len(seeds)} nodes")
        for sid in sorted(seeds)[:5]:
            print(f"  - node={sid} score={float(self.anomaly_scores[sid].item()):.4f}")
        
        # Step 2: Greedy expansion
        expanded_nodes = set(seeds)
        expanded_edges: Set[Tuple[int, int]] = set()
        
        parent = {s: s for s in seeds}
        
        def find(x):
            if parent.get(x, x) != x:
                parent[x] = find(parent[x])
            return parent.get(x, x)
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False
        
        def is_connected():
            return len(set(find(s) for s in seeds)) == 1
        
        budget_used = 0
        
        while not is_connected() and budget_used < self.bridge_budget:
            best_edge = None
            best_reward = -1.0
            
            for node in expanded_nodes:
                for neighbor, eidx in self._get_neighbors(node):
                    src, dst = self._src[eidx], self._dst[eidx]
                    
                    if find(src) == find(dst):
                        continue
                    
                    reward = self._compute_edge_reward(src, dst)
                    if reward > best_reward:
                        best_reward = reward
                        best_edge = (src, dst, eidx)
            
            if best_edge is None:
                print(f"[Connected] No connecting edge found, stopping expansion")
                break
            
            src, dst, _ = best_edge
            expanded_nodes.add(src)
            expanded_nodes.add(dst)
            expanded_edges.add((src, dst))
            union(src, dst)
            budget_used += 1
        
        print(f"[Connected] After expansion: {len(expanded_nodes)} nodes, {len(expanded_edges)} edges")
        
        # Step 3: Evidence completion
        evidence_count = 0
        for node in list(expanded_nodes):
            if float(self.anomaly_scores[node].item()) > anomaly_threshold:
                for neighbor, eidx in self._get_neighbors(node):
                    if float(self._node_rewards[neighbor].item()) > 0.3:
                        if neighbor not in expanded_nodes:
                            evidence_count += 1
                        expanded_nodes.add(neighbor)
                        expanded_edges.add((self._src[eidx], self._dst[eidx]))
        
        print(f"[Connected] Evidence expansion: added {evidence_count} evidence nodes")
        
        # Step 4: Post-pruning
        final_nodes = set(expanded_nodes)
        final_edges = set(expanded_edges)
        pruned_nodes: Set[int] = set()
        pruned_edges: Set[Tuple[int, int]] = set()
        
        changed = True
        prune_rounds = 0
        while changed:
            changed = False
            prune_rounds += 1
            for node in list(final_nodes):
                if node in seeds:
                    continue
                
                degree = sum(1 for s, d in final_edges if s == node or d == node)
                reward = float(self._node_rewards[node].item())
                
                if degree <= 1 and reward < 0.3:
                    final_nodes.discard(node)
                    pruned_nodes.add(node)
                    edges_to_remove = [(s, d) for s, d in final_edges if s == node or d == node]
                    for e in edges_to_remove:
                        final_edges.discard(e)
                        pruned_edges.add(e)
                    changed = True
        
        print(f"[Connected] Pruning done: {prune_rounds} rounds, pruned {len(pruned_nodes)} nodes")
        print(f"[Connected] Final: {len(final_nodes)} nodes, {len(final_edges)} edges")
        
        # Step 5: Build return structure
        layers = ConnectedSubgraphLayers(
            seeds=seeds,
            expanded_nodes=expanded_nodes - seeds,
            expanded_edges=expanded_edges,
            final_nodes=final_nodes,
            final_edges=final_edges,
            pruned_nodes=pruned_nodes,
            pruned_edges=pruned_edges,
        )
        
        # Find attack path
        if len(seeds) >= 2 and final_edges:
            seed_list = sorted(seeds, key=lambda s: -float(self.anomaly_scores[s].item()))
            G = nx.DiGraph()
            for s, d in final_edges:
                G.add_edge(s, d, weight=1.0 - self._compute_edge_reward(s, d) + 0.01)
            
            try:
                layers.attack_path = nx.shortest_path(G, source=seed_list[0], target=seed_list[-1], weight='weight')
                print(f"[Connected] Attack path: {len(layers.attack_path)} nodes")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                print(f"[Connected] No attack path found")
        
        # Build subgraph data
        subgraph_data = {
            'center_node': list(seeds)[0] if seeds else 0,
            'nodes': [],
            'edges': [],
            'attack_path': layers.attack_path,
            'total_anomaly_score': 0.0,
        }
        
        for nid in final_nodes:
            display = self.names_map.get(nid, f"node_{nid}")
            node_data = {
                'node_id': nid,
                'uuid': display,
                'node_type': NODE_TYPE_NAMES.get(self.types_map.get(nid, 0), 'unknown'),
                'name': display,
                'anomaly_score': float(self.anomaly_scores[nid].item()),
                'is_anomaly': float(self.anomaly_scores[nid].item()) > anomaly_threshold,
                'is_seed': nid in seeds,
            }
            if self.cic_scores is not None and self.cic_scores.dim() == 2:
                node_data['cic_scores'] = self.cic_scores[nid].cpu().tolist()
            subgraph_data['nodes'].append(node_data)
        
        path_edges = set(zip(layers.attack_path[:-1], layers.attack_path[1:])) if layers.attack_path else set()
        for src, dst in final_edges:
            edge_data = {
                'src_id': src,
                'dst_id': dst,
                'importance': self._compute_edge_reward(src, dst),
                'is_attack_edge': (src, dst) in path_edges,
                'is_pruned': False,
            }
            subgraph_data['edges'].append(edge_data)
        
        # Add pruned edges for visualization
        for src, dst in pruned_edges:
            edge_data = {
                'src_id': src,
                'dst_id': dst,
                'importance': self._compute_edge_reward(src, dst),
                'is_attack_edge': False,
                'is_pruned': True,
            }
            subgraph_data['edges'].append(edge_data)
        
        if subgraph_data['nodes']:
            subgraph_data['total_anomaly_score'] = sum(
                n['anomaly_score'] for n in subgraph_data['nodes']
            ) / len(subgraph_data['nodes'])
        
        return subgraph_data, layers


def _extract_readable_name(raw: str, node_type: str = '') -> str:
    """Extract readable name from raw entity string"""
    if not raw:
        return ''
    s = str(raw)
    
    # Handle dict-like strings: {'subject': 'firefox'}
    for key in ['subject', 'file', 'netflow', 'memory', 'principal']:
        pattern = f"'{key}': '"
        if pattern in s:
            start = s.find(pattern) + len(pattern)
            end = s.find("'", start)
            if end > start:
                s = s[start:end]
                break
    
    # Clean up common path prefixes
    if '/' in s:
        # Extract meaningful part of path
        # e.g., /usr/bin/firefox -> firefox
        # e.g., /var/log/nginx-*.log -> nginx-*.log
        parts = [p for p in s.split('/') if p]
        if parts:
            # For process names, get the last part
            if node_type == 'subject':
                s = parts[-1]
            # For files, show last 2 parts if path is long
            elif node_type == 'file' and len(parts) > 2:
                s = '/'.join(parts[-2:])
            elif parts:
                s = parts[-1]
    
    # Handle IP:port format for netflow
    if node_type == 'netflow':
        # Keep IP address format as is
        pass
    
    # Truncate if too long
    if len(s) > 20:
        s = s[:17] + '...'
    
    return s


def visualize_connected_subgraph(
    subgraph_data: dict,
    layers: ConnectedSubgraphLayers,
    output_path: str,
    format: str = 'jpeg',
    dpi: int = 1200,
) -> str:
    """
    Paper-quality visualization of connected subgraph.
    
    Style based on KAIROS/APTSHIELD figures:
    - Anomaly nodes: Red filled with black border
    - Normal nodes: White with black border
    - Netflow: Diamond shape with dashed purple border
    - Files: Oval shape
    - Subjects: Rectangle
    - Attack path: Thick red arrows
    """
    try:
        from graphviz import Digraph
    except ImportError:
        print("Warning: graphviz not installed, cannot visualize")
        return ""
    
    dot = Digraph(name="AttackGraph", format=format)
    dot.graph_attr.update({
        'rankdir': 'TB',  # Top to Bottom for better layout
        'dpi': str(dpi),
        'splines': 'ortho',
        'nodesep': '0.5',
        'ranksep': '0.8',
        'fontname': 'Arial',
        'fontsize': '12',
        'bgcolor': 'white',
        'pad': '0.5',
    })
    dot.node_attr.update({
        'fontname': 'Arial',
        'fontsize': '11',
        'margin': '0.1,0.05',
    })
    dot.edge_attr.update({
        'fontname': 'Arial',
        'fontsize': '9',
        'arrowsize': '0.7',
    })
    
    # Node shapes by type (KAIROS paper style)
    SHAPES = {
        'subject': 'box',        # Process: rectangle
        'file': 'ellipse',       # File: oval
        'netflow': 'diamond',    # Network: diamond
        'memory': 'hexagon',
        'principal': 'box',
        'unknown': 'ellipse',
    }
    
    # Attack path edges
    attack_edges = set()
    if layers.attack_path and len(layers.attack_path) >= 2:
        for i in range(len(layers.attack_path) - 1):
            attack_edges.add((layers.attack_path[i], layers.attack_path[i+1]))
    
    # Add FINAL nodes only (skip pruned nodes for cleaner visualization)
    for node in subgraph_data['nodes']:
        nid = node['node_id']
        node_type = node.get('node_type', 'unknown')
        shape = SHAPES.get(node_type, 'ellipse')
        
        # Get readable name
        raw_name = node.get('name', f'node_{nid}')
        label = _extract_readable_name(raw_name, node_type)
        if not label or label.startswith('node_'):
            # Fallback: use short type + id
            label = f"{node_type[:3]}_{nid}"
        
        is_seed = nid in layers.seeds
        is_anomaly = node.get('is_anomaly', False) or is_seed
        
        if is_anomaly:
            # Anomaly nodes: RED filled (like KAIROS paper)
            dot.node(
                name=str(nid),
                label=label,
                shape=shape,
                style='filled,bold',
                fillcolor='#ffcccc',  # Light red fill
                color='#cc0000',       # Red border
                penwidth='2.0',
                fontcolor='#000000',
            )
        else:
            # Normal nodes
            if node_type == 'netflow':
                # Network: purple dashed (KAIROS style)
                dot.node(
                    name=str(nid),
                    label=label,
                    shape=shape,
                    style='dashed',
                    color='#800080',  # Purple
                    penwidth='1.5',
                    fontcolor='#000000',
                )
            else:
                # Files/others: black outline
                dot.node(
                    name=str(nid),
                    label=label,
                    shape=shape,
                    style='solid',
                    color='#000000',
                    penwidth='1.0',
                    fontcolor='#000000',
                )
    
    # Add edges (only non-pruned for cleaner look)
    for edge in subgraph_data['edges']:
        src, dst = edge['src_id'], edge['dst_id']
        is_pruned = edge.get('is_pruned', False)
        
        if is_pruned:
            continue  # Skip pruned edges for cleaner visualization
        
        if (src, dst) in attack_edges:
            # Attack path: thick red
            dot.edge(
                str(src), str(dst),
                style='bold',
                color='#cc0000',
                penwidth='2.0',
            )
        else:
            # Normal edge: black arrow
            dot.edge(
                str(src), str(dst),
                style='solid',
                color='#000000',
                penwidth='1.0',
            )
    
    # Render
    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(output_path))[0]
    full_path = os.path.join(output_dir, filename)
    
    dot.render(full_path, view=False, cleanup=True)
    result_path = f"{full_path}.{format}"
    print(f"[Connected] Saved: {result_path}")
    
    return result_path
