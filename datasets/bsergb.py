"""BS-ERGB training data loader for DSER++.

The BS-ERGB release stores one RGB image per timestamp and one event file for
each adjacent image interval. DSER++ predicts one target timestamp per forward
pass, so a skip-N window is expanded into N independent samples. For example,
skip=3 produces (0, 1, 4), (0, 2, 4), and (0, 3, 4).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from util.event import EventSequence, load_events
from util.utils_func import events_to_channels, process_mask
from util.voxelization import to_voxel_grid


CropBox = Tuple[int, int, int, int]
SampleIndex = Tuple[int, int, int, int, int]


def _natural_name_key(path: Path) -> Tuple[object, ...]:
    """Sort numeric filenames numerically while keeping general names stable."""

    parts: List[object] = []
    token = ""
    is_digit: Optional[bool] = None
    for char in path.name:
        char_is_digit = char.isdigit()
        if is_digit is not None and char_is_digit != is_digit:
            parts.append((0, int(token)) if is_digit else (1, token.lower()))
            token = ""
        token += char
        is_digit = char_is_digit
    if token:
        parts.append((0, int(token)) if is_digit else (1, token.lower()))
    return tuple(parts)


def _list_files(folder: Path, suffixes: Iterable[str]) -> List[Path]:
    suffix_set = {suffix.lower() for suffix in suffixes}
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in suffix_set),
        key=_natural_name_key,
    )


def _load_sequence(
    paths: Sequence[Path],
    height: int,
    width: int,
    crop_box: CropBox,
) -> Optional[EventSequence]:
    """Load, crop, and concatenate an interval of BS-ERGB event files."""

    feature_chunks: List[np.ndarray] = []
    for path in paths:
        features = load_events(str(path), bsergb=True, size=crop_box)
        if features.size:
            feature_chunks.append(features)

    if not feature_chunks:
        return None

    features = np.concatenate(feature_chunks, axis=0)
    if features.shape[0] > 1 and np.any(np.diff(features[:, 2]) < 0):
        features = features[np.argsort(features[:, 2], kind="stable")]
    return EventSequence(features, height, width)


def _voxel_or_zeros(
    sequence: Optional[EventSequence], bins: int, height: int, width: int
) -> torch.Tensor:
    if sequence is None or len(sequence) == 0 or sequence.duration() <= 0:
        return torch.zeros((bins, height, width), dtype=torch.float32)
    return to_voxel_grid(sequence, bins)


def _event_channels_or_zeros(
    sequence: Optional[EventSequence], height: int, width: int
) -> torch.Tensor:
    if sequence is None or len(sequence) == 0:
        return torch.zeros((2, height, width), dtype=torch.float32)
    return events_to_channels(sequence).float()


class BSERGBTrainDataset(Dataset):
    """Create target-specific DSER++ samples from the BS-ERGB training split.

    Expected layout::

        data_root/
          scene_name/
            images/*.png
            events/*.npz

    Images remain in OpenCV BGR channel order to match the existing DSER++
    inference scripts and released checkpoints.
    """

    def __init__(
        self,
        data_root: str | os.PathLike[str],
        bins: int = 8,
        skips: Sequence[int] = (1, 3),
        crop_size: int | Sequence[int] = 256,
        scenes: Optional[Sequence[str]] = None,
        augment: bool = True,
        rotate: bool = True,
        excluded_frame_indices: Optional[Dict[str, Sequence[int]]] = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root).expanduser().resolve()
        self.bins = int(bins)
        self.skips = tuple(sorted({int(skip) for skip in skips}))
        self.augment = bool(augment)
        self.rotate = bool(rotate)
        self.excluded_frame_indices = {
            key: set(value) for key, value in (excluded_frame_indices or {}).items()
        }

        if isinstance(crop_size, int):
            self.crop_height = self.crop_width = crop_size
        else:
            if len(crop_size) != 2:
                raise ValueError("crop_size must be an int or a two-element sequence")
            self.crop_height, self.crop_width = map(int, crop_size)

        if self.bins <= 0:
            raise ValueError("bins must be positive")
        if not self.skips or any(skip <= 0 for skip in self.skips):
            raise ValueError("skips must contain positive integers")
        if self.crop_height <= 0 or self.crop_width <= 0:
            raise ValueError("crop_size must be positive")
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"BS-ERGB root does not exist: {self.data_root}")

        scene_names = list(scenes) if scenes else sorted(
            path.name for path in self.data_root.iterdir() if path.is_dir()
        )
        if not scene_names:
            raise RuntimeError(f"No scene directories found under {self.data_root}")

        self.scene_metadata: List[Dict[str, object]] = []
        self.samples: List[SampleIndex] = []
        for scene_name in scene_names:
            self._register_scene(scene_name)

        if not self.samples:
            raise RuntimeError(
                "No valid BS-ERGB training samples were found. Check the scene "
                "layout, skip values, and excluded frame indices."
            )

    def _register_scene(self, scene_name: str) -> None:
        scene_root = self.data_root / scene_name
        image_folder = scene_root / "images"
        event_folder = scene_root / "events"
        if not image_folder.is_dir() or not event_folder.is_dir():
            raise FileNotFoundError(
                f"Scene {scene_name!r} must contain images/ and events/ directories"
            )

        image_paths = _list_files(image_folder, (".png", ".jpg", ".jpeg"))
        event_paths = _list_files(event_folder, (".npz",))
        if len(image_paths) < 3:
            raise RuntimeError(f"Scene {scene_name!r} contains fewer than three images")
        if len(event_paths) < len(image_paths) - 1:
            raise RuntimeError(
                f"Scene {scene_name!r} has {len(image_paths)} images but only "
                f"{len(event_paths)} event intervals"
            )

        first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Failed to read image: {image_paths[0]}")
        height, width = first.shape[:2]
        if height < self.crop_height or width < self.crop_width:
            raise ValueError(
                f"Scene {scene_name!r} has resolution {width}x{height}, smaller "
                f"than crop {self.crop_width}x{self.crop_height}"
            )

        scene_index = len(self.scene_metadata)
        self.scene_metadata.append(
            {
                "name": scene_name,
                "images": image_paths,
                "events": event_paths,
                "height": height,
                "width": width,
            }
        )

        excluded = self.excluded_frame_indices.get(scene_name, set())
        for skip in self.skips:
            endpoint_offset = skip + 1
            for start in range(len(image_paths) - endpoint_offset):
                end = start + endpoint_offset
                if any(frame in excluded for frame in range(start, end + 1)):
                    continue
                for target in range(start + 1, end):
                    self.samples.append((scene_index, start, target, end, skip))

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_image(path: Path) -> torch.Tensor:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0

    def _choose_crop(self, height: int, width: int) -> CropBox:
        if self.augment:
            top = random.randint(0, height - self.crop_height)
            left = random.randint(0, width - self.crop_width)
        else:
            top = (height - self.crop_height) // 2
            left = (width - self.crop_width) // 2
        return top, left, top + self.crop_height, left + self.crop_width

    @staticmethod
    def _apply_spatial_transform(
        images: torch.Tensor,
        voxels: torch.Tensor,
        mask: torch.Tensor,
        vertical_flip: bool,
        horizontal_flip: bool,
        rotations: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if vertical_flip:
            images = torch.flip(images, dims=(-2,))
            voxels = torch.flip(voxels, dims=(-2,))
            mask = torch.flip(mask, dims=(-2,))
        if horizontal_flip:
            images = torch.flip(images, dims=(-1,))
            voxels = torch.flip(voxels, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        if rotations:
            images = torch.rot90(images, rotations, dims=(-2, -1))
            voxels = torch.rot90(voxels, rotations, dims=(-2, -1))
            mask = torch.rot90(mask, rotations, dims=(-2, -1))
        return images.contiguous(), voxels.contiguous(), mask.contiguous()

    def __getitem__(self, index: int) -> Dict[str, object]:
        scene_index, start, target, end, skip = self.samples[index]
        metadata = self.scene_metadata[scene_index]
        image_paths = metadata["images"]
        event_paths = metadata["events"]
        height = int(metadata["height"])
        width = int(metadata["width"])

        crop_box = self._choose_crop(height, width)
        top, left, bottom, right = crop_box
        selected_images = torch.cat(
            [
                self._read_image(image_paths[frame])[:, top:bottom, left:right]
                for frame in (start, target, end)
            ],
            dim=0,
        )

        events_0t = _load_sequence(
            event_paths[start:target], self.crop_height, self.crop_width, crop_box
        )
        events_t1 = _load_sequence(
            event_paths[target:end], self.crop_height, self.crop_width, crop_box
        )

        event_counts = torch.cat(
            [
                _event_channels_or_zeros(events_0t, self.crop_height, self.crop_width),
                _event_channels_or_zeros(events_t1, self.crop_height, self.crop_width),
            ],
            dim=0,
        )
        event_mask = process_mask(event_counts)

        voxel_0t = _voxel_or_zeros(
            events_0t, self.bins, self.crop_height, self.crop_width
        )
        voxel_t1 = _voxel_or_zeros(
            events_t1, self.bins, self.crop_height, self.crop_width
        )
        events_1t = None if events_t1 is None else events_t1.copy().reverse()
        voxel_1t = _voxel_or_zeros(
            events_1t, self.bins, self.crop_height, self.crop_width
        )
        voxels = torch.cat((voxel_0t, voxel_t1, voxel_1t), dim=0).float()

        if self.augment:
            vertical_flip = random.random() < 0.5
            horizontal_flip = random.random() < 0.5
            if self.rotate and self.crop_height == self.crop_width:
                rotations = random.randrange(4)
            elif self.rotate:
                rotations = random.randrange(2) * 2
            else:
                rotations = 0
            selected_images, voxels, event_mask = self._apply_spatial_transform(
                selected_images,
                voxels,
                event_mask,
                vertical_flip,
                horizontal_flip,
                rotations,
            )

        return {
            "imgs": selected_images,
            "gt": selected_images[3:6],
            "voxels": voxels,
            "mask": event_mask.float(),
            "scene": str(metadata["name"]),
            "start_index": start,
            "target_index": target,
            "end_index": end,
            "skip": skip,
        }


def build_bsergb_train_dataset(
    data_root: str | os.PathLike[str],
    bins: int = 8,
    skips: Sequence[int] = (1, 3),
    crop_size: int | Sequence[int] = 256,
    scenes: Optional[Sequence[str]] = None,
    augment: bool = True,
) -> BSERGBTrainDataset:
    """Convenience builder used by train_bsergb.py."""

    return BSERGBTrainDataset(
        data_root=data_root,
        bins=bins,
        skips=skips,
        crop_size=crop_size,
        scenes=scenes,
        augment=augment,
    )
