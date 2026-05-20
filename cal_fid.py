import argparse
import json
from pathlib import Path


def calculate_fid(original_dir, modified_dir, batch_size=50, dims=2048, num_workers=1):
    import torch
    from pytorch_fid import fid_score

    original_dir = Path(original_dir)
    modified_dir = Path(modified_dir)
    if not original_dir.is_dir():
        raise ValueError(f"Original image directory does not exist: {original_dir}")
    if not modified_dir.is_dir():
        raise ValueError(f"Modified image directory does not exist: {modified_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return fid_score.calculate_fid_given_paths(
        paths=[str(original_dir), str(modified_dir)],
        batch_size=batch_size,
        device=device,
        dims=dims,
        num_workers=num_workers,
    )


def write_result(output_path, result):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FID between two direct image folders.")
    parser.add_argument("--original-dir", "--original", dest="original_dir", required=True)
    parser.add_argument("--modified-dir", "--modified", dest="modified_dir", required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dims", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--output", default="evaluation_results/fid.json")
    return parser.parse_args()


def main():
    args = parse_args()
    fid_value = calculate_fid(
        original_dir=args.original_dir,
        modified_dir=args.modified_dir,
        batch_size=args.batch_size,
        dims=args.dims,
        num_workers=args.num_workers,
    )
    result = {
        "metric": "FID",
        "original_dir": args.original_dir,
        "modified_dir": args.modified_dir,
        "fid": float(fid_value),
        "batch_size": args.batch_size,
        "dims": args.dims,
        "num_workers": args.num_workers,
    }
    write_result(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
