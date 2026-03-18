import argparse
import json
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
    parser = argparse.ArgumentParser(description="Few-shot target-city fine-tuning for TAPSTAR")
    parser.add_argument("--source_city", type=str, required=True)
    parser.add_argument("--target_city", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--descriptor_path", type=str, default="artifacts/city_descriptors_dim16.npz")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--target_days", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def default_seq_len(city: str) -> int:
    return 12 if city.lower() in {"metr-la", "pems-bay"} else 6


def few_shot_ranges(total_time: int, target_days: int, steps_per_day: int):
    train_end = min(total_time, target_days * steps_per_day)
    val_end = min(total_time, train_end + steps_per_day)
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


def configure_trainable_parameters(model: TAPSTAR):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.tapr.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = True
    if model.city_prompt is not None:
        for param in model.city_prompt.parameters():
            param.requires_grad = True


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    seq_len = default_seq_len(args.target_city)
    steps_per_day = get_steps_per_day(args.target_city)
    output_dir = Path(args.output_root) / f"{args.source_city}_to_{args.target_city}" / "finetune"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("tapstar_finetune", str(output_dir / "train.log"))

    series, adj = load_city_series(args.data_root, args.target_city)
    features = build_features(series, args.target_city)
    (train_start, train_end), (val_start, val_end), (test_start, test_end) = few_shot_ranges(
        features.shape[0], args.target_days, steps_per_day
    )
    mean, std = compute_speed_stats(features, train_start, train_end)
    city_desc_table, city_idx, city_desc_dim = build_city_desc(args.target_city, args.descriptor_path, device)

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
        else Path(args.output_root) / args.source_city / "tapstar_pretrain" / "tapstar_source_best.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Source pre-training checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")["model"]
    model.load_state_dict(state, strict=False)
    configure_trainable_parameters(model)
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    patience = 0
    best_path = output_dir / "tapstar_target_best.pt"

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
        val_loss, val_metrics = evaluate(model, val_loader, city_desc_table, mean, std, args.target_city, device)
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
            patience = 0
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, best_path)
            logger.info("Saved best checkpoint to %s", best_path)
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    best_state = torch.load(best_path, map_location="cpu")["model"]
    model.load_state_dict(best_state, strict=False)
    test_loss, test_metrics = evaluate(model, test_loader, city_desc_table, mean, std, args.target_city, device)
    logger.info("Final test loss %.6f | metrics %s", test_loss, test_metrics)
    summary = {"test_loss": test_loss, "metrics": test_metrics}
    with open(output_dir / "final_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
