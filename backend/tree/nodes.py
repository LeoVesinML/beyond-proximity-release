"""TreeNode dataclass for the hierarchical semantic scene tree."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class NodeType(str, Enum):
    ROOT   = "root"
    ZONE   = "zone"
    REGION = "region"
    LEAF   = "leaf"


@dataclass
class TreeNode:
    node_id:   str
    node_type: NodeType
    name:      str
    summary:   str
    depth:     int = 0

    parent_id:   Optional[str] = None
    children_ids: List[str]    = field(default_factory=list)
    view_ids:    List[str]     = field(default_factory=list)

    # Spatial statistics (populated after construction)
    centroid:          List[float]      = field(default_factory=lambda: [0.0, 0.0, 0.0])
    camera_positions:  List[List[float]] = field(default_factory=list)

    # Optional geometry hint for ablation study
    geometry_assisted: bool = True

    def compute_spatial_stats(self) -> None:
        """Recompute centroid from stored camera_positions."""
        if not self.camera_positions:
            return
        arr = np.array(self.camera_positions)
        self.centroid = arr.mean(axis=0).tolist()

    def to_dict(self) -> dict:
        return {
            "node_id":          self.node_id,
            "node_type":        self.node_type.value,
            "name":             self.name,
            "summary":          self.summary,
            "depth":            self.depth,
            "parent_id":        self.parent_id,
            "children_ids":     self.children_ids,
            "view_ids":         self.view_ids,
            "centroid":         self.centroid,
            "camera_positions": self.camera_positions,
            "geometry_assisted": self.geometry_assisted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TreeNode":
        n = cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            name=data["name"],
            summary=data["summary"],
            depth=data.get("depth", 0),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            view_ids=data.get("view_ids", []),
            centroid=data.get("centroid", [0.0, 0.0, 0.0]),
            camera_positions=data.get("camera_positions", []),
            geometry_assisted=data.get("geometry_assisted", True),
        )
        return n
