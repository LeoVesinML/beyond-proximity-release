#!/usr/bin/env python3
"""Minimal Python 3 reader for official ScanNet v4 ``.sens`` files."""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image


COLOR_COMPRESSION_TYPES = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
DEPTH_COMPRESSION_TYPES = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


def read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, received {len(data)}")
    return data


def read_scalar(handle: BinaryIO, fmt: str) -> int | float:
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, read_exact(handle, size))[0]


def read_matrix(handle: BinaryIO) -> np.ndarray:
    return np.frombuffer(read_exact(handle, 16 * 4), dtype="<f4").reshape(4, 4).copy()


@dataclass(frozen=True)
class SensorFrame:
    index: int
    camera_to_world: np.ndarray
    timestamp_color: int
    timestamp_depth: int
    color_offset: int
    color_size_bytes: int
    depth_offset: int
    depth_size_bytes: int


@dataclass(frozen=True)
class SensorHeader:
    version: int
    sensor_name: str
    intrinsic_color: np.ndarray
    extrinsic_color: np.ndarray
    intrinsic_depth: np.ndarray
    extrinsic_depth: np.ndarray
    color_compression_type: str
    depth_compression_type: str
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    depth_shift: float
    frame_count: int


@dataclass(frozen=True)
class DecodedSensorFrame:
    color: Image.Image
    depth_meters: np.ndarray
    camera_to_world: np.ndarray
    timestamp_color: int
    timestamp_depth: int


class SensorDataIndex:
    """Indexed access to ScanNet sensor frames without loading all payloads."""

    def __init__(self, path: Path, header: SensorHeader, frames: list[SensorFrame]) -> None:
        self.path = path
        self.header = header
        self.frames = frames

    @classmethod
    def scan(cls, path: Path) -> "SensorDataIndex":
        path = path.resolve()
        with path.open("rb") as handle:
            version = int(read_scalar(handle, "I"))
            if version != 4:
                raise ValueError(f"Unsupported ScanNet .sens version {version}; expected 4")
            name_length = int(read_scalar(handle, "Q"))
            sensor_name = read_exact(handle, name_length).decode("utf-8", errors="replace")
            intrinsic_color = read_matrix(handle)
            extrinsic_color = read_matrix(handle)
            intrinsic_depth = read_matrix(handle)
            extrinsic_depth = read_matrix(handle)
            color_code = int(read_scalar(handle, "i"))
            depth_code = int(read_scalar(handle, "i"))
            if color_code not in COLOR_COMPRESSION_TYPES:
                raise ValueError(f"Unknown ScanNet color compression code: {color_code}")
            if depth_code not in DEPTH_COMPRESSION_TYPES:
                raise ValueError(f"Unknown ScanNet depth compression code: {depth_code}")
            color_width = int(read_scalar(handle, "I"))
            color_height = int(read_scalar(handle, "I"))
            depth_width = int(read_scalar(handle, "I"))
            depth_height = int(read_scalar(handle, "I"))
            depth_shift = float(read_scalar(handle, "f"))
            if depth_shift <= 0:
                raise ValueError(f"Invalid ScanNet depth_shift: {depth_shift}")
            frame_count = int(read_scalar(handle, "Q"))

            frames: list[SensorFrame] = []
            for index in range(frame_count):
                camera_to_world = read_matrix(handle)
                timestamp_color = int(read_scalar(handle, "Q"))
                timestamp_depth = int(read_scalar(handle, "Q"))
                color_size = int(read_scalar(handle, "Q"))
                depth_size = int(read_scalar(handle, "Q"))
                color_offset = handle.tell()
                depth_offset = color_offset + color_size
                handle.seek(color_size + depth_size, 1)
                frames.append(
                    SensorFrame(
                        index=index,
                        camera_to_world=camera_to_world,
                        timestamp_color=timestamp_color,
                        timestamp_depth=timestamp_depth,
                        color_offset=color_offset,
                        color_size_bytes=color_size,
                        depth_offset=depth_offset,
                        depth_size_bytes=depth_size,
                    )
                )

        header = SensorHeader(
            version=version,
            sensor_name=sensor_name,
            intrinsic_color=intrinsic_color,
            extrinsic_color=extrinsic_color,
            intrinsic_depth=intrinsic_depth,
            extrinsic_depth=extrinsic_depth,
            color_compression_type=COLOR_COMPRESSION_TYPES[color_code],
            depth_compression_type=DEPTH_COMPRESSION_TYPES[depth_code],
            color_width=color_width,
            color_height=color_height,
            depth_width=depth_width,
            depth_height=depth_height,
            depth_shift=depth_shift,
            frame_count=frame_count,
        )
        return cls(path, header, frames)

    def _decode_color(self, payload: bytes) -> Image.Image:
        compression = self.header.color_compression_type
        if compression in {"jpeg", "png"}:
            with Image.open(io.BytesIO(payload)) as image:
                return image.convert("RGB")
        if compression == "raw":
            expected = self.header.color_width * self.header.color_height * 3
            if len(payload) != expected:
                raise ValueError(f"Raw color payload has {len(payload)} bytes; expected {expected}")
            array = np.frombuffer(payload, dtype=np.uint8).reshape(
                self.header.color_height, self.header.color_width, 3
            )
            return Image.fromarray(array, mode="RGB")
        raise ValueError(f"Unsupported color compression: {compression}")

    def _decode_depth(self, payload: bytes) -> np.ndarray:
        compression = self.header.depth_compression_type
        if compression == "zlib_ushort":
            payload = zlib.decompress(payload)
        elif compression != "raw_ushort":
            raise ValueError(f"Unsupported depth compression: {compression}")
        expected_values = self.header.depth_width * self.header.depth_height
        depth = np.frombuffer(payload, dtype="<u2")
        if depth.size != expected_values:
            raise ValueError(f"Depth payload has {depth.size} values; expected {expected_values}")
        depth_meters = depth.reshape(self.header.depth_height, self.header.depth_width).astype(np.float32)
        depth_meters /= self.header.depth_shift
        return depth_meters

    def decode(self, frame_index: int, *, resize_color_to_depth: bool = True) -> DecodedSensorFrame:
        frame = self.frames[frame_index]
        with self.path.open("rb") as handle:
            handle.seek(frame.color_offset)
            color_payload = read_exact(handle, frame.color_size_bytes)
            handle.seek(frame.depth_offset)
            depth_payload = read_exact(handle, frame.depth_size_bytes)
        color = self._decode_color(color_payload)
        if resize_color_to_depth and color.size != (self.header.depth_width, self.header.depth_height):
            color = color.resize(
                (self.header.depth_width, self.header.depth_height),
                resample=Image.Resampling.BILINEAR,
            )
        return DecodedSensorFrame(
            color=color,
            depth_meters=self._decode_depth(depth_payload),
            camera_to_world=frame.camera_to_world.copy(),
            timestamp_color=frame.timestamp_color,
            timestamp_depth=frame.timestamp_depth,
        )

    def metadata(self) -> dict[str, object]:
        header = self.header
        return {
            "version": header.version,
            "sensor_name": header.sensor_name,
            "frame_count": header.frame_count,
            "color_compression_type": header.color_compression_type,
            "depth_compression_type": header.depth_compression_type,
            "color_width": header.color_width,
            "color_height": header.color_height,
            "depth_width": header.depth_width,
            "depth_height": header.depth_height,
            "depth_shift": header.depth_shift,
            "intrinsic_color": header.intrinsic_color.tolist(),
            "extrinsic_color": header.extrinsic_color.tolist(),
            "intrinsic_depth": header.intrinsic_depth.tolist(),
            "extrinsic_depth": header.extrinsic_depth.tolist(),
        }
