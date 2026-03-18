import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import TrafficDataset, build_features, collate_fn, compute_speed_stats, load_city_series
from masking import temporal_block_mask
from models import TAPRPretrainer
from utils import get_logger, get_steps_per_day, load_city_descriptors, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="TAPR residual pre-training")
    parser.add_argument("--source_city", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--descriptor_path", type=str, default="artifacts/city_descriptors_dim16.npz")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--node_mask_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def default_seq_len(city: str) -> int:
    return 12 if city.lower() in {"metr-la", "pems-bay"} else 6


def split_ranges(total_time: int):
    train_end = int(total_time * 0.7)
    val_end = int(total_time * 0.8)
    return (0, train_end), (train_end, val_end)


def build_city_desc(city: str, descriptor_path: str, device: torch.device):
    city_ids, desc_table, city_to_idx = load_city_descriptors(descriptor_path)
    if city not in city_to_idx:
        raise ValueError(f"City {city} not found in descriptors: {city_ids}")
    desc = torch.from_numpy(desc_table).to(device)
    return desc, city_to_idx[city], desc_table.shape[1]


def apply_mask_token(x: torch.Tensor, mask: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    masked = x.clone()
    mask_f = mask.float()
    masked[..., 0] = masked[..., 0] * (1.0 - mask_f) + mask_token.view(1, 1, 1) * mask_f
    return masked


def sample_mask(x: torch.Tensor, mask_ratio: float, node_mask_ratio: float) -> torch.Tensor:
    temporal_mask = temporal_block_mask(
        batch_size=x.shape[0],
        seq_len=x.shape[1],
        num_nodes=x.shape[2],
        mask_ratio=mask_ratio,
        block_len_min=2,
        block_len_max=6,
        device=x.device,
    )
    if node_mask_ratio > 0:
        node_mask = (torch.rand((x.shape[0], x.shape[2]), device=x.device) < node_mask_ratio)
        node_mask = node_mask[:, None, :].expand(x.shape[0], x.shape[1], x.shape[2])
        return temporal_mask | node_mask
    return temporal_mask


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    seq_len = default_seq_len(args.source_city)
    steps_per_day = get_steps_per_day(args.source_city)

    output_dir = Path(args.output_root) / args.source_city / "tapr_pretrain"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("tapr_pretrain", str(output_dir / "train.log"))

    series, adj = load_city_series(args.data_root, args.source_city)
    features = build_features(series, args.source_city)
    (train_start, train_end), (val_start, val_end) = split_ranges(features.shape[0])
    mean, std = compute_speed_stats(features, train_start, train_end)

    city_desc_table, city_idx, city_desc_dim = build_city_desc(args.source_city, args.descriptor_path, device)
    train_set = TrafficDataset(
        features,
        adj,
        history_len=seq_len,
        pred_len=seq_len,
        start_index=train_start,
        end_index=train_end,
        stride=1,
        mean=mean,
        std=std,
        city_id=city_idx,
        return_city_id=True,
    )
    val_set = TrafficDataset(
        features,
        adj,
        history_len=seq_len,
        pred_len=seq_len,
        start_index=val_start,
        end_index=val_end,
        stride=1,
        mean=mean,
        std=std,
        city_id=city_idx,
        return_city_id=True,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = TAPRPretrainer(
        seq_len=seq_len,
        tod_size=steps_per_day,
        num_prototypes=8,
        num_heads=8,
        use_city_prompt=True,
        city_desc_dim=city_desc_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path = output_dir / "tapr_pretrain_best.pt"
    last_path = output_dir / "tapr_pretrain_last.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x, _, _, city_ids in train_loader:
            x = x.to(device)
            city_desc = city_desc_table[city_ids.to(device)]
            mask = sample_mask(x, args.mask_ratio, args.node_mask_ratio)
            masked_x = apply_mask_token(x, mask, model.base.mask_token)
            coarse, delta_hat, _ = model(masked_x, mae_mask=mask, city_desc=city_desc)
            gt_speed = x[..., 0]
            delta_target = gt_speed - coarse.detach()
            loss = F.smooth_l1_loss(delta_hat[mask], delta_target[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_count += 1

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for x, _, _, city_ids in val_loader:
                x = x.to(device)
                city_desc = city_desc_table[city_ids.to(device)]
                mask = sample_mask(x, args.mask_ratio, args.node_mask_ratio)
                masked_x = apply_mask_token(x, mask, model.base.mask_token)
                coarse, delta_hat, _ = model(masked_x, mae_mask=mask, city_desc=city_desc)
                gt_speed = x[..., 0]
                delta_target = gt_speed - coarse.detach()
                loss = F.smooth_l1_loss(delta_hat[mask], delta_target[mask])
                val_loss_sum += loss.item()
                val_count += 1

        avg_train = train_loss_sum / max(1, train_count)
        avg_val = val_loss_sum / max(1, val_count)
        logger.info("Epoch %d | train %.6f | val %.6f", epoch, avg_train, avg_val)

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, last_path)
        if avg_val < best_val:
            best_val = avg_val
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, best_path)
            logger.info("Saved best checkpoint to %s", best_path)


if __name__ == "__main__":
    main()

