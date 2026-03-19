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
python TAPSTAR/run_tapstar.py --source_city metr-la --target_city pems-bay
```

Generate city descriptors only:

```bash
python TAPSTAR/compute_city_descriptors.py --source_city metr-la --target_city pems-bay
```

Residual pre-training:

```bash
python TAPSTAR/pretrain_residual.py --source_city metr-la
```

Source supervised pre-training:

```bash
python TAPSTAR/pretrain_source.py --source_city metr-la
```

Target-city fine-tuning:

```bash
python TAPSTAR/finetune_target.py --source_city metr-la --target_city pems-bay
```
