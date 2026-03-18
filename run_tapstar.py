import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="One-click TAPSTAR pipeline runner")
    parser.add_argument("--source_city", type=str, required=True)
    parser.add_argument("--target_city", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--descriptor_path", type=str, default="artifacts/city_descriptors_dim16.npz")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def run_command(cmd, workdir: Path):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(workdir), check=True)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    python_bin = sys.executable

    residual_ckpt = root / args.output_root / args.source_city / "tapr_pretrain" / "tapr_pretrain_best.pt"
    source_ckpt = root / args.output_root / args.source_city / "tapstar_pretrain" / "tapstar_source_best.pt"
    final_metrics_path = (
        root
        / args.output_root
        / f"{args.source_city}_to_{args.target_city}"
        / "finetune"
        / "final_metrics.json"
    )

    run_command(
        [
            python_bin,
            "pretrain_residual.py",
            "--source_city",
            args.source_city,
            "--data_root",
            args.data_root,
            "--descriptor_path",
            args.descriptor_path,
            "--output_root",
            args.output_root,
            "--device",
            args.device,
        ],
        root,
    )

    run_command(
        [
            python_bin,
            "pretrain_source.py",
            "--source_city",
            args.source_city,
            "--data_root",
            args.data_root,
            "--descriptor_path",
            args.descriptor_path,
            "--output_root",
            args.output_root,
            "--checkpoint",
            str(residual_ckpt),
            "--device",
            args.device,
        ],
        root,
    )

    run_command(
        [
            python_bin,
            "finetune_target.py",
            "--source_city",
            args.source_city,
            "--target_city",
            args.target_city,
            "--data_root",
            args.data_root,
            "--descriptor_path",
            args.descriptor_path,
            "--output_root",
            args.output_root,
            "--checkpoint",
            str(source_ckpt),
            "--device",
            args.device,
        ],
        root,
    )

    if not final_metrics_path.exists():
        raise FileNotFoundError(f"Final metrics file not found: {final_metrics_path}")

    with open(final_metrics_path, "r", encoding="utf-8") as fp:
        summary = json.load(fp)

    print("\n[FINAL RESULT]")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
