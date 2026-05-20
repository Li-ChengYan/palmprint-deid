import argparse
import json
from pathlib import Path

from evaluation_utils import pair_image_directories


def load_lpips_tensor(path, resize_max_side=2000):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if max(image.size) > resize_max_side:
        image.thumbnail((resize_max_side, resize_max_side), resample=Image.Resampling.LANCZOS)
    tensor = torch.tensor(np.array(image)).float()
    return tensor.permute(2, 0, 1) / 127.5 - 1.0


def summarize_scores(scores):
    import numpy as np

    return {
        "mean": float(np.mean(scores)) if scores else None,
        "std": float(np.std(scores)) if scores else None,
        "count": len(scores),
    }


def calculate_lpips(original_dir, modified_dir, suffix="_de-id", net="vgg", resize_max_side=2000):
    import lpips
    import torch
    from tqdm import tqdm

    pairs, stats = pair_image_directories(original_dir, modified_dir, suffix=suffix)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net=net).to(device)

    scores = []
    errors = []
    mismatches = 0
    for original_path, modified_path in tqdm(pairs, desc="LPIPS"):
        try:
            original = load_lpips_tensor(original_path, resize_max_side=resize_max_side)
            modified = load_lpips_tensor(modified_path, resize_max_side=resize_max_side)
            if original.shape != modified.shape:
                mismatches += 1
                continue
            with torch.no_grad():
                score = loss_fn(original.unsqueeze(0).to(device), modified.unsqueeze(0).to(device))
            scores.append(float(score.cpu().numpy().reshape(-1)[0]))
        except Exception as error:
            errors.append(f"{original_path} -> {modified_path}: {error}")

    result = dict(stats)
    result["mismatch"] = mismatches
    result["errors"] = len(errors)
    result["error_messages"] = errors
    result["metric"] = "LPIPS"
    result["net"] = net
    result["suffix"] = suffix
    result["scores"] = summarize_scores(scores)
    return result


def write_result(output_path, result):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate LPIPS for direct image folders.")
    parser.add_argument("--original-dir", "--original", dest="original_dir", required=True)
    parser.add_argument("--modified-dir", "--modified", dest="modified_dir", required=True)
    parser.add_argument("--suffix", default="_de-id")
    parser.add_argument("--net", default="vgg", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--resize-max-side", type=int, default=2000)
    parser.add_argument("--output", default="evaluation_results/lpips.json")
    return parser.parse_args()


def main():
    args = parse_args()
    result = calculate_lpips(
        original_dir=args.original_dir,
        modified_dir=args.modified_dir,
        suffix=args.suffix,
        net=args.net,
        resize_max_side=args.resize_max_side,
    )
    result["original_dir"] = args.original_dir
    result["modified_dir"] = args.modified_dir
    write_result(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
