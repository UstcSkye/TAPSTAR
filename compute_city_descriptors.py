import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import build_features, load_city_series
from utils import get_steps_per_day


def parse_args():
    parser = argparse.ArgumentParser(description="Generate city descriptors for TAPSTAR")
    parser.add_argument("--source_city", type=str, required=True)
    parser.add_argument("--target_city", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--out_path", type=str, default="artifacts/city_descriptors_dim16.npz")
    parser.add_argument("--target_days", type=int, default=3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def get_split_range(total_time: int, split: str, val_ratio: float, test_ratio: float, target_days: int | None, steps_per_day: int):
    if target_days is not None and target_days > 0:
        cutoff = min(total_time, target_days * steps_per_day)
        if split == "train":
            return 0, cutoff
        if split == "val":
            return cutoff, cutoff
        if split == "test":
            return cutoff, total_time
        raise ValueError(f"Unknown split: {split}")

    train_end = int(total_time * (1.0 - val_ratio - test_ratio))
    val_end = int(total_time * (1.0 - test_ratio))
    if split == "train":
        return 0, train_end
    if split == "val":
        return train_end, val_end
    if split == "test":
        return val_end, total_time
    raise ValueError(f"Unknown split: {split}")


def infer_week_steps(time_index: np.ndarray) -> int:
    max_idx = float(np.max(time_index))
    if max_idx >= 1500:
        return 2016
    if max_idx >= 900:
        return 1008
    return 0


def max_missing_run(mask_1d: np.ndarray) -> int:
    longest = 0
    current = 0
    for val in mask_1d:
        if val:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def compute_descriptor(series: np.ndarray, city: str, start: int, end: int, steps_per_day: int, eps: float = 1e-6) -> np.ndarray:
    seg = series[start:end]
    speed = seg[:, :, 0].astype(np.float32)
    t_len, num_nodes = speed.shape

    missing_mask = ~np.isfinite(speed)
    missing_ratio = float(missing_mask.mean(axis=0).mean())

    speed_nan = speed.copy()
    speed_nan[missing_mask] = np.nan

    mean_node = np.nanmean(speed_nan, axis=0)
    std_node = np.nanstd(speed_nan, axis=0)
    min_node = np.nanmin(speed_nan, axis=0)
    max_node = np.nanmax(speed_nan, axis=0)
    p10_node = np.nanpercentile(speed_nan, 10, axis=0)
    p25_node = np.nanpercentile(speed_nan, 25, axis=0)
    p50_node = np.nanpercentile(speed_nan, 50, axis=0)
    p75_node = np.nanpercentile(speed_nan, 75, axis=0)
    p90_node = np.nanpercentile(speed_nan, 90, axis=0)
    peak_valley_ratio_node = p90_node / (p10_node + eps)
    mean_abs_diff_node = np.nanmean(np.abs(np.diff(speed_nan, axis=0)), axis=0)

    features = build_features(seg, city)
    tod = features[:, 0, 1]
    tod_idx = np.clip((tod * steps_per_day).astype(np.int64), 0, steps_per_day - 1)
    tod_var_nodes = []
    for n in range(num_nodes):
        valid = ~np.isnan(speed_nan[:, n])
        if not np.any(valid):
            tod_var_nodes.append(0.0)
            continue
        counts = np.bincount(tod_idx[valid], minlength=steps_per_day).astype(np.float32)
        sums = np.bincount(tod_idx[valid], weights=speed_nan[valid, n], minlength=steps_per_day).astype(np.float32)
        curve = sums / (counts + eps)
        tod_var_nodes.append(float(np.var(curve)))
    tod_var = float(np.mean(tod_var_nodes))

    max_runs = [max_missing_run(missing_mask[:, n]) for n in range(num_nodes)]
    max_missing_run_mean = float(np.mean(max_runs))

    mean_abs_corr = 0.0
    if num_nodes > 1 and t_len > 1:
        speed_corr = speed_nan
        col_mean = np.nanmean(speed_corr, axis=0)
        speed_filled = np.where(np.isnan(speed_corr), col_mean, speed_corr)
        corr = np.corrcoef(speed_filled.T)
        mean_abs_corr = float(np.mean(np.abs(corr[np.triu_indices_from(corr, k=1)])))

    descriptor = np.array(
        [
            float(np.nanmean(mean_node)),
            float(np.nanmean(std_node)),
            float(np.nanmean(min_node)),
            float(np.nanmean(max_node)),
            float(np.nanmean(p10_node)),
            float(np.nanmean(p25_node)),
            float(np.nanmean(p50_node)),
            float(np.nanmean(p75_node)),
            float(np.nanmean(p90_node)),
            float(np.nanmean(peak_valley_ratio_node)),
            float(np.nanmean(mean_abs_diff_node)),
            float(tod_var),
            float(missing_ratio),
            float(max_missing_run_mean),
            float(num_nodes),
            float(mean_abs_corr),
        ],
        dtype=np.float32,
    )
    return descriptor


def main():
    args = parse_args()
    np.random.seed(args.seed)

    cities = [args.source_city, args.target_city]
    descriptors = []
    for city in cities:
        series, _ = load_city_series(args.data_root, city)
        steps_per_day = get_steps_per_day(city)
        if city == args.target_city:
            start, end = get_split_range(
                series.shape[0],
                split="train",
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                target_days=args.target_days,
                steps_per_day=steps_per_day,
            )
        else:
            start, end = get_split_range(
                series.shape[0],
                split="train",
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                target_days=None,
                steps_per_day=steps_per_day,
            )
        descriptors.append(compute_descriptor(series, city, start, end, steps_per_day))

    descriptors_raw = np.stack(descriptors, axis=0).astype(np.float32)
    mu_source = descriptors_raw[0].copy()
    sigma_source = np.maximum(np.zeros_like(mu_source, dtype=np.float32) + 1e-6, np.ones_like(mu_source, dtype=np.float32) * 1e-6)
    descriptors_norm = (descriptors_raw - mu_source[None, :]) / sigma_source[None, :]
    # For a single-source public release, source normalization uses the source descriptor itself.
    # This keeps the file format consistent with the training code.

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        city_ids=np.array(cities, dtype=object),
        descriptors_raw=descriptors_raw,
        mu_source=mu_source.astype(np.float32),
        sigma_source=sigma_source.astype(np.float32),
        descriptors_norm=descriptors_norm.astype(np.float32),
    )

    print(json.dumps({"out_path": str(out_path), "cities": cities, "descriptor_dim": int(descriptors_raw.shape[1])}, indent=2))


if __name__ == "__main__":
    main()
