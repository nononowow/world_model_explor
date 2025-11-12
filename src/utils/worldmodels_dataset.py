"""
Dataset utilities for the official World Models rollouts.

The original World Models release (https://worldmodels.github.io) provides
pre-recorded trajectories for two benchmark environments:

* `CarRacing-v0` (continuous control, RGB observations)
* `VizDoomTakeCover-v0` (discrete control, grayscale observations)

Each environment ships with four NumPy arrays:
  - observations.npy
  - actions.npy
  - rewards.npy
  - terminals.npy

The data is stored in flattened rollout order. This module reconstructs
episode boundaries using the `terminals` array and exposes a PyTorch Dataset
that yields complete episodes as tuples of
`(observations, actions, rewards, dones)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "worldmodels"


@dataclass(frozen=True)
class EpisodeSlice:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


DATA_KEYS = ("observations.npy", "actions.npy", "rewards.npy", "terminals.npy")
SUPPORTED_ENVS = ("car_racing", "doom_take_cover")


class WorldModelsEpisodeDataset(Dataset):
    """
    World Models rollouts grouped by episode.

    Parameters
    ----------
    env : str
        Environment identifier. One of ``"car_racing"`` or ``"doom_take_cover"``.
    root : Path-like, optional
        Directory that contains the unpacked `.npy` files (defaults to
        ``data/raw/worldmodels/<env>`` relative to the repo root).
    normalize_observations : bool, optional
        If ``True`` observations are scaled to ``[0, 1]`` when the source dtype
        is integer. Floating point observations are left untouched. Defaults to
        ``True``.
    channels_first : bool, optional
        Return images as ``(T, C, H, W)`` instead of the stored ``(T, H, W, C)``
        format. Defaults to ``True`` which is convenient for PyTorch models.
    mmap : bool, optional
        Load arrays in memory-mapped read-only mode. This avoids loading the
        entire dataset into RAM upfront. Defaults to ``True``.
    device : torch.device or str, optional
        If provided, move tensors to the given device on ``__getitem__``.
    """

    def __init__(
        self,
        env: str = "car_racing",
        root: Optional[Path] = None,
        *,
        normalize_observations: bool = True,
        channels_first: bool = True,
        mmap: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        if env not in SUPPORTED_ENVS:
            raise ValueError(f"Unsupported environment '{env}'. Expected one of {SUPPORTED_ENVS}.")

        self.env = env
        self.root = Path(root) if root is not None else DEFAULT_ROOT / env
        self.normalize_observations = normalize_observations
        self.channels_first = channels_first
        self.mmap = mmap
        self.device = device

        self._observations = self._load_array("observations.npy")
        self._actions = self._load_array("actions.npy")
        self._rewards = self._load_array("rewards.npy")
        self._terminals = self._load_array("terminals.npy").astype(bool)

        self._validate_shapes()
        self._episode_slices = self._compute_episode_slices(self._terminals)

    # ------------------------------------------------------------------ #
    # PyTorch Dataset API
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._episode_slices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        slice_ = self._episode_slices[index]

        obs = self._extract_observations(slice_)
        actions = self._actions[slice_.start : slice_.end]
        rewards = self._rewards[slice_.start : slice_.end]
        dones = self._terminals[slice_.start : slice_.end]

        obs_tensor = self._to_tensor(obs, dtype=torch.float32)
        action_tensor = self._to_tensor(actions, dtype=torch.float32)
        reward_tensor = self._to_tensor(rewards, dtype=torch.float32)
        done_tensor = self._to_tensor(dones, dtype=torch.bool)

        return obs_tensor, action_tensor, reward_tensor, done_tensor

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def episode_lengths(self) -> List[int]:
        return [slice_.length for slice_ in self._episode_slices]

    @property
    def num_steps(self) -> int:
        return int(self._terminals.shape[0])

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_array(self, filename: str) -> np.ndarray:
        if filename not in DATA_KEYS:
            raise ValueError(f"Unknown dataset key '{filename}'.")

        path = self.root / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected file '{path}' not found. "
                "Please download the dataset with scripts/download_worldmodels_data.py."
            )

        mmap_mode = "r" if self.mmap else None
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)

    def _validate_shapes(self) -> None:
        num_steps = self._terminals.shape[0]
        for name, array in (
            ("observations", self._observations),
            ("actions", self._actions),
            ("rewards", self._rewards),
        ):
            if array.shape[0] != num_steps:
                raise ValueError(
                    f"Mismatch between terminals (len={num_steps}) and {name} (len={array.shape[0]})."
                )

    @staticmethod
    def _compute_episode_slices(terminals: np.ndarray) -> List[EpisodeSlice]:
        if terminals.ndim != 1:
            raise ValueError("Terminals array must be 1-D.")

        terminal_indices = np.flatnonzero(terminals) + 1
        if terminal_indices.size == 0 or terminal_indices[-1] != len(terminals):
            terminal_indices = np.append(terminal_indices, len(terminals))

        starts = np.concatenate(([0], terminal_indices[:-1]))
        slices = [EpisodeSlice(int(start), int(end)) for start, end in zip(starts, terminal_indices)]

        # Filter out empty slices (can happen if consecutive terminal flags appear)
        return [slice_ for slice_ in slices if slice_.length > 0]

    def _extract_observations(self, slice_: EpisodeSlice) -> np.ndarray:
        obs = self._observations[slice_.start : slice_.end]

        # Handle grayscale arrays stored as (T, H, W)
        if obs.ndim == 3:
            obs = obs[..., None]

        if self.normalize_observations and not np.issubdtype(obs.dtype, np.floating):
            obs = obs.astype(np.float32) / 255.0
        else:
            obs = obs.astype(np.float32, copy=False)

        if self.channels_first:
            obs = np.transpose(obs, (0, 3, 1, 2))

        return obs

    def _to_tensor(self, array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.as_tensor(array, dtype=dtype)
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor


def split_episodes(lengths: Sequence[int], max_steps: Optional[int] = None) -> List[List[int]]:
    """
    Utility to split episode indices into mini-batches constrained by ``max_steps``.

    Useful when you want to iterate over multiple episodes but keep the total
    number of steps per batch under control (e.g. to fit into GPU memory).
    """

    if max_steps is None:
        return [[idx] for idx in range(len(lengths))]

    batches: List[List[int]] = []
    current: List[int] = []
    remaining = max_steps

    for idx, length in enumerate(lengths):
        if length > max_steps:
            raise ValueError(
                f"Episode {idx} has {length} steps which exceeds max_steps={max_steps}."
            )

        if length > remaining and current:
            batches.append(current)
            current = []
            remaining = max_steps

        current.append(idx)
        remaining -= length

    if current:
        batches.append(current)

    return batches


def episode_collate_fn(batch: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Collate function that keeps the variable-length episodes intact.

    This simply wraps the list batch into tuple-of-lists for convenience.
    """

    observations, actions, rewards, dones = zip(*batch)
    return list(observations), list(actions), list(rewards), list(dones)

