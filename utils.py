import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_logger(name: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_city_time_step(city: str) -> int:
    city = city.lower()
    if city in {"metr-la", "pems-bay"}:
        return 5
    if city in {"chengdu", "shenzhen"}:
        return 10
    return 5


def get_steps_per_day(city: str) -> int:
    return int((24 * 60) / get_city_time_step(city))


def get_horizon_minutes(city: str) -> List[int]:
    city = city.lower()
    if city in {"metr-la", "pems-bay"}:
        return [5, 15, 30]
    if city in {"chengdu", "shenzhen"}:
        return [10, 30, 60]
    step = get_city_time_step(city)
    return [step, step * 3, step * 6]


def horizon_minutes_to_steps(horizon_minutes: List[int], time_step: int) -> List[int]:
    return [max(1, int(round(minutes / time_step))) for minutes in horizon_minutes]


def init_metric_tracker(horizon_minutes: List[int]) -> Dict:
    size = len(horizon_minutes)
    return {
        "horizon_minutes": horizon_minutes,
        "mae_sum": [0.0] * size,
        "mape_sum": [0.0] * size,
        "count": [0] * size,
    }


def update_metrics(
    tracker: Dict,
    preds: torch.Tensor,
    labels: torch.Tensor,
    mean: float,
    std: float,
    time_step: int,
) -> None:
    preds = preds * std + mean
    labels = labels * std + mean
    eps = 1e-5
    horizon_steps = horizon_minutes_to_steps(tracker["horizon_minutes"], time_step)

    for i, step in enumerate(horizon_steps):
        idx = step - 1
        if idx >= preds.shape[1]:
            continue
        pred = preds[:, idx, :, 0]
        label = labels[:, idx, :, 0]
        diff = torch.abs(pred - label)
        tracker["mae_sum"][i] += diff.sum().item()
        tracker["mape_sum"][i] += (diff / torch.clamp(label.abs(), min=eps)).sum().item()
        tracker["count"][i] += label.numel()


def summarize_metrics(tracker: Dict) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for i, minutes in enumerate(tracker["horizon_minutes"]):
        count = max(1, tracker["count"][i])
        metrics[f"MAE@{minutes}"] = tracker["mae_sum"][i] / count
        metrics[f"MAPE@{minutes}"] = tracker["mape_sum"][i] / count * 100.0
    return metrics


def average_mae(metrics: Dict[str, float]) -> float:
    maes = [value for key, value in metrics.items() if key.startswith("MAE@")]
    return float(np.mean(maes)) if maes else float("inf")


def load_city_descriptors(path: str):
    data = np.load(path, allow_pickle=True)
    city_ids = [str(x) for x in data["city_ids"].tolist()]
    desc_table = data["descriptors_norm"].astype(np.float32)
    city_to_idx = {city: idx for idx, city in enumerate(city_ids)}
    return city_ids, desc_table, city_to_idx

