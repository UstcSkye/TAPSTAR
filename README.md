# TAPSTAR

## Data layout

```text
data/
  metr-la/
    dataset.npy
    matrix.npy
  pems-bay/
    dataset.npy
    matrix.npy
  shenzhen/
    dataset.npy
    matrix.npy
  chengdu/
    dataset.npy
    matrix.npy
```

If city prompt is enabled, place the descriptor file at:

```text
artifacts/city_descriptors_dim16.npz
```

## Run

One-click pipeline:

```bash
python tapstar_release/run_tapstar.py --source_city metr-la --target_city pems-bay
```

Residual pre-training:

```bash
python tapstar_release/pretrain_residual.py --source_city metr-la
```

Source supervised pre-training:

```bash
python tapstar_release/pretrain_source.py --source_city metr-la
```

Target-city fine-tuning:

```bash
python tapstar_release/finetune_target.py --source_city metr-la --target_city pems-bay
```
