from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_steps_per_day


def load_city_series(data_root: str, city: str) -> Tuple[np.ndarray, np.ndarray]:
    series = np.load(f"{data_root}/{city}/dataset.npy")
    adj = np.load(f"{data_root}/{city}/matrix.npy")
    if series.ndim != 3:
        raise ValueError("dataset.npy should have shape [T, N, C].")
    return series.astype(np.float32), adj.astype(np.float32)


def _infer_week_steps(time_index: np.ndarray) -> int:
    max_idx = float(np.max(time_index))
    if max_idx >= 1500:
        return 2016
    if max_idx >= 900:
        return 1008
    return 0


def build_features(series: np.ndarray, city: str, steps_per_day_override: Optional[int] = None) -> np.ndarray:
    total_time, num_nodes, channels = series.shape
    speed = series[:, :, 0].astype(np.float32)
    steps_per_day = (
        int(steps_per_day_override)
        if steps_per_day_override is not None and int(steps_per_day_override) > 0
        else get_steps_per_day(city)
    )

    if channels >= 2:
        tod = series[:, :, 1].astype(np.float32)
    else:
        tod = (np.arange(total_time, dtype=np.float32) % steps_per_day) / float(steps_per_day)
        tod = np.repeat(tod[:, None], num_nodes, axis=1)
    tod = np.clip(tod, 0.0, 1.0 - 1e-6)

    if channels >= 4:
        time_index = series[:, 0, 3].astype(np.float32)
        week_steps = _infer_week_steps(time_index)
        if week_steps == 0:
            week_steps = steps_per_day * 7
        day_steps = week_steps // 7
        day_index = (time_index // day_steps).astype(np.int64) % 7
    else:
        day_index = (np.arange(total_time) // steps_per_day).astype(np.int64) % 7
    dow = (day_index.astype(np.float32) / 7.0).reshape(-1, 1)
    dow = np.repeat(dow, num_nodes, axis=1)

    return np.stack([speed, tod, dow], axis=-1).astype(np.float32)


def compute_speed_stats(series: np.ndarray, start: int, end: int) -> Tuple[float, float]:
    if end <= start:
        raise ValueError("Invalid range for speed statistics.")
    speed = series[start:end, :, 0]
    mean = float(speed.mean())
    std = max(float(speed.std()), 1e-6)
    return mean, std


class TrafficDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        adj: np.ndarray,
        history_len: int,
        pred_len: int,
        start_index: int,
        end_index: int,
        stride: int,
        mean: float,
        std: float,
        city_id: Optional[object] = None,
        return_city_id: bool = False,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.pred_len = pred_len
        self.stride = stride
        self.adj = torch.from_numpy(adj.astype(np.float32))
        self.city_id = city_id
        self.return_city_id = return_city_id

        features = features.copy()
        features[:, :, 0] = (features[:, :, 0] - mean) / std
        self.features = features

        last_start = end_index - history_len - pred_len
        self.indices = list(range(max(0, start_index), last_start + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        x = torch.from_numpy(self.features[t : t + self.history_len])
        y = torch.from_numpy(self.features[t + self.history_len : t + self.history_len + self.pred_len, :, 0:1])
        if self.return_city_id:
            return x, y, self.adj, self.city_id
        return x, y, self.adj


def collate_fn(batch):
    if len(batch[0]) == 4:
        xs, ys, adjs, city_ids = zip(*batch)
        return torch.stack(xs, 0), torch.stack(ys, 0), adjs[0], torch.as_tensor(city_ids, dtype=torch.long)
    xs, ys, adjs = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), adjs[0]
