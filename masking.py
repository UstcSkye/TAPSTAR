import torch


@torch.no_grad()
def temporal_block_mask(
    batch_size: int,
    seq_len: int,
    num_nodes: int,
    mask_ratio: float,
    block_len_min: int = 2,
    block_len_max: int = 6,
    seed: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate temporal block masks with shape [B, T, N]."""
    if device is None:
        device = torch.device("cpu")

    if mask_ratio <= 0:
        return torch.zeros((batch_size, seq_len, num_nodes), dtype=torch.bool, device=device)
    if mask_ratio >= 1:
        return torch.ones((batch_size, seq_len, num_nodes), dtype=torch.bool, device=device)

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    mask = torch.zeros((batch_size, seq_len, num_nodes), dtype=torch.bool, device=device)
    target = max(1, int(round(mask_ratio * seq_len)))
    masked = torch.zeros((batch_size, num_nodes), dtype=torch.int64, device=device)
    time_index = torch.arange(seq_len, device=device).view(1, seq_len, 1)

    max_iters = max(4, seq_len * 2)
    iters = 0
    while torch.any(masked < target) and iters < max_iters:
        iters += 1
        block_len = torch.randint(
            low=block_len_min,
            high=block_len_max + 1,
            size=(batch_size, num_nodes),
            device=device,
            generator=generator,
        )
        block_len = torch.clamp(block_len, max=seq_len)
        max_start = torch.clamp(seq_len - block_len, min=0)
        start = torch.floor(
            torch.rand((batch_size, num_nodes), device=device, generator=generator)
            * (max_start + 1).float()
        ).long()

        block = (time_index >= start.unsqueeze(1)) & (time_index < (start + block_len).unsqueeze(1))
        mask |= block
        masked = mask.sum(dim=1)

    if torch.any(masked < target):
        deficit = (target - masked).clamp(min=0)
        for b in range(batch_size):
            for n in range(num_nodes):
                if deficit[b, n] > 0:
                    idx = torch.randperm(seq_len, device=device)[: deficit[b, n]]
                    mask[b, idx, n] = True

    return mask

