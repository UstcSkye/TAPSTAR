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
    parser.add_argument("--lambda_rec0", type=float, default=1.0)
    parser.add_argument("--lambda_pred_max", type=float, default=1.0)
    parser.add_argument("--pred_warmup_epochs", type=int, default=30)
    parser.add_argument("--pred_ramp_epochs", type=int, default=0)
    parser.add_argument("--lambda_cons", type=float, default=0.1)
    parser.add_argument("--lambda_balance", type=float, default=0.001)
    parser.add_argument("--lambda_entropy", type=float, default=0.001)
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


def get_pred_weight(epoch: int, args) -> float:
    warmup = max(0, int(args.pred_warmup_epochs))
    ramp = max(0, int(args.pred_ramp_epochs))
    if epoch <= warmup:
        scale = 0.0
    elif ramp > 0:
        scale = min(1.0, float(epoch - warmup) / float(ramp))
    else:
        scale = 1.0
    return float(args.lambda_pred_max) * scale


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
        node_mask = torch.rand((x.shape[0], x.shape[2]), device=x.device) < node_mask_ratio
        node_mask = node_mask[:, None, :].expand(x.shape[0], x.shape[1], x.shape[2])
        temporal_mask = temporal_mask | node_mask
    return temporal_mask


def aggregate_prototype_tokens(z: torch.Tensor, assignment: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = torch.einsum("bhtkn,btnd->bhkd", assignment, z)
    den = assignment.sum(dim=(2, 4))
    tokens = num / (den.unsqueeze(-1) + eps)
    return tokens.mean(dim=1)


def prototype_usage_regularizer(assignment: torch.Tensor, eps: float = 1e-6):
    usage = assignment.mean(dim=(0, 2, 4)).mean(dim=0)
    target = torch.full_like(usage, 1.0 / usage.numel())
    balance = ((usage - target) ** 2).mean()
    probs = assignment.permute(0, 1, 2, 4, 3)
    entropy = -(probs * (probs + eps).log()).sum(dim=-1).mean()
    return balance, entropy, usage.min(), usage.max()


def evaluate(model, loader, city_desc_table, device, args):
    loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    pred_loss_fn = torch.nn.L1Loss()
    model.eval()

    total = {
        "res": 0.0,
        "rec0": 0.0,
        "pred": 0.0,
        "cons": 0.0,
        "balance": 0.0,
        "entropy": 0.0,
        "total": 0.0,
        "steps": 0,
    }

    with torch.no_grad():
        for x, y, _, city_ids in loader:
            x = x.to(device)
            y = y.to(device)
            city_desc = city_desc_table[city_ids.to(device)]
            pred_weight = get_pred_weight(args.current_epoch, args)

            mask1 = sample_mask(x, args.mask_ratio, args.node_mask_ratio)
            mask2 = sample_mask(x, args.mask_ratio, args.node_mask_ratio)
            mask_f = mask1.float()

            x_masked = apply_mask_token(x, mask1, model.base.mask_token)
            coarse, delta_hat, _, z1, assignment1 = model(
                x_masked,
                mae_mask=mask1,
                city_desc=city_desc,
                return_assignment=True,
            )
            gt_speed = x[..., 0]
            delta_target = gt_speed - coarse.detach()

            res_loss = (loss_fn(delta_hat, delta_target) * mask_f).sum() / mask_f.sum().clamp_min(1.0)
            rec0_loss = (loss_fn(coarse, gt_speed) * mask_f).sum() / mask_f.sum().clamp_min(1.0)

            pred_loss = torch.zeros((), device=device)
            if pred_weight > 0.0:
                _, _, pred = model.base(x)
                pred_loss = pred_loss_fn(pred, y) * pred_weight

            x_masked2 = apply_mask_token(x, mask2, model.base.mask_token)
            _, _, _, z2, assignment2 = model(
                x_masked2,
                mae_mask=mask2,
                city_desc=city_desc,
                return_assignment=True,
            )
            c1 = aggregate_prototype_tokens(z1, assignment1)
            c2 = aggregate_prototype_tokens(z2, assignment2)
            c1 = c1 / (c1.norm(dim=-1, keepdim=True) + 1e-6)
            c2 = c2 / (c2.norm(dim=-1, keepdim=True) + 1e-6)
            cons_loss = (1.0 - (c1 * c2).sum(dim=-1)).mean()

            balance_loss, entropy_loss, _, _ = prototype_usage_regularizer(assignment1)
            loss = (
                res_loss
                + args.lambda_rec0 * rec0_loss
                + pred_loss
                + args.lambda_cons * cons_loss
                + args.lambda_balance * balance_loss
                + args.lambda_entropy * entropy_loss
            )

            total["res"] += res_loss.item()
            total["rec0"] += rec0_loss.item()
            total["pred"] += pred_loss.item()
            total["cons"] += cons_loss.item()
            total["balance"] += balance_loss.item()
            total["entropy"] += entropy_loss.item()
            total["total"] += loss.item()
            total["steps"] += 1

    denom = max(1, total["steps"])
    return {key: value / denom for key, value in total.items() if key != "steps"}


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
    loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    pred_loss_fn = torch.nn.L1Loss()

    best_val = float("inf")
    best_path = output_dir / "tapr_pretrain_best.pt"
    last_path = output_dir / "tapr_pretrain_last.pt"

    for epoch in range(1, args.epochs + 1):
        args.current_epoch = epoch
        pred_weight = get_pred_weight(epoch, args)
        model.train()
        totals = {
            "res": 0.0,
            "rec0": 0.0,
            "pred": 0.0,
            "cons": 0.0,
            "balance": 0.0,
            "entropy": 0.0,
            "total": 0.0,
            "steps": 0,
        }

        for x, y, _, city_ids in train_loader:
            x = x.to(device)
            y = y.to(device)
            city_desc = city_desc_table[city_ids.to(device)]
            mask1 = sample_mask(x, args.mask_ratio, args.node_mask_ratio)
            mask2 = sample_mask(x, args.mask_ratio, args.node_mask_ratio)
            mask_f = mask1.float()

            x_masked = apply_mask_token(x, mask1, model.base.mask_token)
            coarse, delta_hat, _, z1, assignment1 = model(
                x_masked,
                mae_mask=mask1,
                city_desc=city_desc,
                return_assignment=True,
            )
            gt_speed = x[..., 0]
            delta_target = gt_speed - coarse.detach()
            res_loss = (loss_fn(delta_hat, delta_target) * mask_f).sum() / mask_f.sum().clamp_min(1.0)
            rec0_loss = (loss_fn(coarse, gt_speed) * mask_f).sum() / mask_f.sum().clamp_min(1.0)

            pred_loss = torch.zeros((), device=device)
            if pred_weight > 0.0:
                _, _, pred = model.base(x)
                pred_loss = pred_loss_fn(pred, y) * pred_weight

            x_masked2 = apply_mask_token(x, mask2, model.base.mask_token)
            _, _, _, z2, assignment2 = model(
                x_masked2,
                mae_mask=mask2,
                city_desc=city_desc,
                return_assignment=True,
            )
            c1 = aggregate_prototype_tokens(z1, assignment1)
            c2 = aggregate_prototype_tokens(z2, assignment2)
            c1 = c1 / (c1.norm(dim=-1, keepdim=True) + 1e-6)
            c2 = c2 / (c2.norm(dim=-1, keepdim=True) + 1e-6)
            cons_loss = (1.0 - (c1 * c2).sum(dim=-1)).mean()
            balance_loss, entropy_loss, usage_min, usage_max = prototype_usage_regularizer(assignment1)

            loss = (
                res_loss
                + args.lambda_rec0 * rec0_loss
                + pred_loss
                + args.lambda_cons * cons_loss
                + args.lambda_balance * balance_loss
                + args.lambda_entropy * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals["res"] += res_loss.item()
            totals["rec0"] += rec0_loss.item()
            totals["pred"] += pred_loss.item()
            totals["cons"] += cons_loss.item()
            totals["balance"] += balance_loss.item()
            totals["entropy"] += entropy_loss.item()
            totals["total"] += loss.item()
            totals["steps"] += 1

        denom = max(1, totals["steps"])
        train_stats = {key: value / denom for key, value in totals.items() if key != "steps"}
        val_stats = evaluate(model, val_loader, city_desc_table, device, args)

        logger.info(
            "Epoch %d | lambda_pred %.4f | train total %.6f | val total %.6f | "
            "train(res %.6f rec0 %.6f pred %.6f cons %.6f bal %.6f ent %.6f) | "
            "val(res %.6f rec0 %.6f pred %.6f cons %.6f bal %.6f ent %.6f)",
            epoch,
            pred_weight,
            train_stats["total"],
            val_stats["total"],
            train_stats["res"],
            train_stats["rec0"],
            train_stats["pred"],
            train_stats["cons"],
            train_stats["balance"],
            train_stats["entropy"],
            val_stats["res"],
            val_stats["rec0"],
            val_stats["pred"],
            val_stats["cons"],
            val_stats["balance"],
            val_stats["entropy"],
        )
        logger.info("Epoch %d | usage min/max %.6f / %.6f", epoch, usage_min.item(), usage_max.item())

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, last_path)
        if epoch > args.pred_warmup_epochs and val_stats["total"] < best_val:
            best_val = val_stats["total"]
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, best_path)
            logger.info("Saved best checkpoint to %s", best_path)


if __name__ == "__main__":
    main()
