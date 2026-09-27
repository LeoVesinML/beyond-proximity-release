"""Read/write camera poses in Nerfstudio transforms.json format."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class FrameEntry:
    file_path: str
    transform_matrix: List[List[float]]   # 4×4 camera-to-world
    view_id: str = ""
    depth_file_path: Optional[str] = None

    def camera_position(self) -> List[float]:
        """Return [x, y, z] translation from the 4th column of the matrix."""
        m = self.transform_matrix
        return [m[0][3], m[1][3], m[2][3]]

    def rotation_matrix_3x3(self) -> List[List[float]]:
        """Return the upper-left 3×3 rotation sub-matrix."""
        m = self.transform_matrix
        return [[m[r][c] for c in range(3)] for r in range(3)]

    def to_dict(self) -> dict:
        d: dict = {
            "file_path": self.file_path,
            "transform_matrix": self.transform_matrix,
        }
        if self.view_id:
            d["view_id"] = self.view_id
        if self.depth_file_path:
            d["depth_file_path"] = self.depth_file_path
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FrameEntry":
        return cls(
            file_path=data["file_path"],
            transform_matrix=data["transform_matrix"],
            view_id=data.get("view_id", ""),
            depth_file_path=data.get("depth_file_path"),
        )


@dataclass
class NerfstudioScene:
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    w: int
    h: int
    camera_model: str = "OPENCV"
    frames: List[FrameEntry] = field(default_factory=list)

    def add_frame(self, frame: FrameEntry) -> None:
        self.frames.append(frame)

    def get_frame(self, view_id: str) -> Optional[FrameEntry]:
        for f in self.frames:
            if f.view_id == view_id:
                return f
        return None

    def to_dict(self) -> dict:
        return {
            "fl_x": self.fl_x,
            "fl_y": self.fl_y,
            "cx": self.cx,
            "cy": self.cy,
            "w": self.w,
            "h": self.h,
            "camera_model": self.camera_model,
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NerfstudioScene":
        scene = cls(
            fl_x=data["fl_x"],
            fl_y=data["fl_y"],
            cx=data["cx"],
            cy=data["cy"],
            w=data["w"],
            h=data["h"],
            camera_model=data.get("camera_model", "OPENCV"),
        )
        for f in data.get("frames", []):
            scene.frames.append(FrameEntry.from_dict(f))
        return scene


def save_transforms(scene: NerfstudioScene, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(scene.to_dict(), fh, indent=2)


def load_transforms(path: str) -> NerfstudioScene:
    with open(path) as fh:
        data = json.load(fh)
    return NerfstudioScene.from_dict(data)


def get_or_create_transforms(
    path: str,
    fl_x: float = 800.0,
    fl_y: float = 800.0,
    cx: float = 400.0,
    cy: float = 300.0,
    w: int = 800,
    h: int = 600,
) -> NerfstudioScene:
    if Path(path).exists():
        return load_transforms(path)
    scene = NerfstudioScene(fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy, w=w, h=h)
    save_transforms(scene, path)
    return scene
