import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import TrafficDataset, build_features, collate_fn, compute_speed_stats, load_city_series
from models import TAPSTAR
from utils import (
    average_mae,
    get_city_time_step,
    get_horizon_minutes,
    get_logger,
    get_steps_per_day,
    init_metric_tracker,
    load_city_descriptors,
    set_seed,
    summarize_metrics,
    update_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Source-city supervised pre-training for TAPSTAR")
    parser.add_argument("--source_city", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--descriptor_path", type=str, default="artifacts/city_descriptors_dim16.npz")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def default_seq_len(city: str) -> int:
    return 12 if city.lower() in {"metr-la", "pems-bay"} else 6


def split_ranges(total_time: int):
    train_end = int(total_time * 0.7)
    val_end = int(total_time * 0.8)
    return (0, train_end), (train_end, val_end), (val_end, total_time)


def build_city_desc(city: str, descriptor_path: str, device: torch.device):
    city_ids, desc_table, city_to_idx = load_city_descriptors(descriptor_path)
    if city not in city_to_idx:
        raise ValueError(f"City {city} not found in descriptors: {city_ids}")
    desc = torch.from_numpy(desc_table).to(device)
    return desc, city_to_idx[city], desc_table.shape[1]


def evaluate(model, loader, city_desc_table, mean, std, city, device):
    tracker = init_metric_tracker(get_horizon_minutes(city))
    total_loss = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for x, y, _, city_ids in loader:
            x = x.to(device)
            y = y.to(device)
            city_desc = city_desc_table[city_ids.to(device)]
            pred = model(x, city_desc=city_desc)
            loss = F.l1_loss(pred, y)
            total_loss += loss.item()
            count += 1
            update_metrics(tracker, pred, y, mean, std, get_city_time_step(city))
    return total_loss / max(1, count), summarize_metrics(tracker)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    seq_len = default_seq_len(args.source_city)
    steps_per_day = get_steps_per_day(args.source_city)
    output_dir = Path(args.output_root) / args.source_city / "tapstar_pretrain"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("tapstar_source_pretrain", str(output_dir / "train.log"))

    series, adj = load_city_series(args.data_root, args.source_city)
    features = build_features(series, args.source_city)
    (train_start, train_end), (val_start, val_end), (test_start, test_end) = split_ranges(features.shape[0])
    mean, std = compute_speed_stats(features, train_start, train_end)
    city_desc_table, city_idx, city_desc_dim = build_city_desc(args.source_city, args.descriptor_path, device)

    train_set = TrafficDataset(features, adj, seq_len, seq_len, train_start, train_end, 1, mean, std, city_idx, True)
    val_set = TrafficDataset(features, adj, seq_len, seq_len, val_start, val_end, 1, mean, std, city_idx, True)
    test_set = TrafficDataset(features, adj, seq_len, seq_len, test_start, test_end, 1, mean, std, city_idx, True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = TAPSTAR(
        seq_len=seq_len,
        horizon=seq_len,
        tod_size=steps_per_day,
        num_prototypes=8,
        num_heads=8,
        use_city_prompt=True,
        city_desc_dim=city_desc_dim,
    ).to(device)

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(args.output_root) / args.source_city / "tapr_pretrain" / "tapr_pretrain_best.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Residual pre-training checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded residual checkpoint from %s", ckpt_path)
    if missing:
        logger.info("Missing keys: %s", missing)
    if unexpected:
        logger.info("Unexpected keys: %s", unexpected)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_path = output_dir / "tapstar_source_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        count = 0
        for x, y, _, city_ids in train_loader:
            x = x.to(device)
            y = y.to(device)
            city_desc = city_desc_table[city_ids.to(device)]
            pred = model(x, city_desc=city_desc)
            loss = F.l1_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            count += 1

        avg_train = train_loss / max(1, count)
        val_loss, val_metrics = evaluate(model, val_loader, city_desc_table, mean, std, args.source_city, device)
        val_score = average_mae(val_metrics)
        logger.info(
            "Epoch %d | train %.6f | val %.6f | val_avg_mae %.6f",
            epoch,
            avg_train,
            val_loss,
            val_score,
        )

        if val_score < best_val:
            best_val = val_score
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, best_path)
            logger.info("Saved best checkpoint to %s", best_path)

    best_state = torch.load(best_path, map_location="cpu")["model"]
    model.load_state_dict(best_state, strict=False)
    test_loss, test_metrics = evaluate(model, test_loader, city_desc_table, mean, std, args.source_city, device)
    logger.info("Final test loss %.6f | metrics %s", test_loss, test_metrics)


if __name__ == "__main__":
    main()

