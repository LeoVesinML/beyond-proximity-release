"""Persist and load tree nodes to/from JSON files on disk."""
import json
from pathlib import Path
from typing import Dict, List, Optional

from backend.tree.nodes import TreeNode, NodeType


def _tree_dir(scene_dir: str) -> Path:
    return Path(scene_dir) / "tree"


def _manifest_path(scene_dir: str) -> Path:
    return _tree_dir(scene_dir) / "manifest.json"


def _node_path(scene_dir: str, node_id: str) -> Path:
    return _tree_dir(scene_dir) / f"node_{node_id}.json"


def save_node(scene_dir: str, node: TreeNode) -> None:
    d = _tree_dir(scene_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(_node_path(scene_dir, node.node_id), "w") as fh:
        json.dump(node.to_dict(), fh, indent=2)
    _update_manifest(scene_dir, node)


def load_node(scene_dir: str, node_id: str) -> Optional[TreeNode]:
    path = _node_path(scene_dir, node_id)
    if not path.exists():
        return None
    with open(path) as fh:
        return TreeNode.from_dict(json.load(fh))


def load_all_nodes(scene_dir: str) -> Dict[str, TreeNode]:
    d = _tree_dir(scene_dir)
    if not d.exists():
        return {}
    nodes: Dict[str, TreeNode] = {}
    for p in d.glob("node_*.json"):
        with open(p) as fh:
            n = TreeNode.from_dict(json.load(fh))
        nodes[n.node_id] = n
    return nodes


def get_root_node_id(scene_dir: str) -> Optional[str]:
    mp = _manifest_path(scene_dir)
    if not mp.exists():
        return None
    with open(mp) as fh:
        data = json.load(fh)
    return data.get("root_id")


def _update_manifest(scene_dir: str, node: TreeNode) -> None:
    mp = _manifest_path(scene_dir)
    if mp.exists():
        with open(mp) as fh:
            data = json.load(fh)
    else:
        data = {"root_id": None, "node_ids": []}

    if node.node_id not in data["node_ids"]:
        data["node_ids"].append(node.node_id)
    if node.node_type == NodeType.ROOT or node.parent_id is None:
        data["root_id"] = node.node_id

    with open(mp, "w") as fh:
        json.dump(data, fh, indent=2)


def to_d3_tree(scene_dir: str) -> Optional[dict]:
    """Return a D3-compatible nested dict for the full tree."""
    nodes = load_all_nodes(scene_dir)
    if not nodes:
        return None
    root_id = get_root_node_id(scene_dir)
    if root_id is None or root_id not in nodes:
        # Fall back to node with no parent
        for n in nodes.values():
            if n.parent_id is None:
                root_id = n.node_id
                break
    if root_id is None:
        return None
    return _build_d3(nodes, root_id)


def _build_d3(nodes: Dict[str, TreeNode], node_id: str) -> dict:
    n = nodes[node_id]
    d: dict = {
        "id": n.node_id,
        "name": n.name,
        "type": n.node_type.value,
        "summary": n.summary,
        "view_ids": n.view_ids,
        "centroid": n.centroid,
    }
    if n.children_ids:
        d["children"] = [
            _build_d3(nodes, cid) for cid in n.children_ids if cid in nodes
        ]
    return d
