import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluation_utils import pair_image_directories


def load_image(path, resize_max_side=2000, resize_min_side=161, resize_small_side=200):
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if max(image.size) > resize_max_side:
        image.thumbnail((resize_max_side, resize_max_side), resample=Image.Resampling.LANCZOS)
    if min(image.size) < resize_min_side:
        image = image.resize((resize_small_side, resize_small_side), resample=Image.Resampling.LANCZOS)
    return np.array(image)


def compute_psnr(original, modified):
    import skimage.metrics

    return skimage.metrics.peak_signal_noise_ratio(original, modified, data_range=255)


def compute_ssim(original, modified):
    import skimage.metrics

    return skimage.metrics.structural_similarity(
        original,
        modified,
        data_range=255,
        channel_axis=-1 if original.ndim == 3 else None,
    )


def compute_ms_ssim(original, modified):
    import torch
    import piq

    original_tensor = torch.tensor(original, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    modified_tensor = torch.tensor(modified, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    return piq.multi_scale_ssim(original_tensor, modified_tensor, data_range=1.0).item()


def summarize_metric(values):
    import numpy as np

    return {
        "mean": float(np.nanmean(values)) if values else None,
        "std": float(np.nanstd(values)) if values else None,
        "count": len(values),
    }


def calculate_similarity(
    original_dir,
    modified_dir,
    suffix="_de-id",
    workers=None,
    resize_max_side=2000,
    resize_min_side=161,
    resize_small_side=200,
):
    pairs, stats = pair_image_directories(original_dir, modified_dir, suffix=suffix)
    stats = defaultdict(int, stats)
    stats["mismatch"] += 0
    stats["errors"] += 0
    metrics = defaultdict(list)

    def process_pair(original_path, modified_path):
        try:
            original = load_image(
                original_path,
                resize_max_side=resize_max_side,
                resize_min_side=resize_min_side,
                resize_small_side=resize_small_side,
            )
            modified = load_image(
                modified_path,
                resize_max_side=resize_max_side,
                resize_min_side=resize_min_side,
                resize_small_side=resize_small_side,
            )
            if original.shape != modified.shape:
                return {"mismatch": 1}, None
            return {}, {
                "PSNR": compute_psnr(original, modified),
                "SSIM": compute_ssim(original, modified),
                "MS-SSIM": compute_ms_ssim(original, modified),
            }
        except Exception as error:
            return {"errors": 1, "error_messages": [f"{original_path} -> {modified_path}: {error}"]}, None

    worker_count = workers or os.cpu_count() or 1
    error_messages = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(process_pair, original_path, modified_path) for original_path, modified_path in pairs]
        for future in futures:
            stat_result, metric_result = future.result()
            for key, value in stat_result.items():
                if key == "error_messages":
                    error_messages.extend(value)
                else:
                    stats[key] += value
            if metric_result:
                for key, value in metric_result.items():
                    metrics[key].append(value)

    result = dict(stats)
    result["metrics"] = {
        metric: summarize_metric(metrics.get(metric, [])) for metric in ["PSNR", "SSIM", "MS-SSIM"]
    }
    result["error_messages"] = error_messages
    result["suffix"] = suffix
    return result


def write_result(output_path, result):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate PSNR, SSIM, and MS-SSIM for direct image folders.")
    parser.add_argument("--original-dir", "--original", dest="original_dir", required=True)
    parser.add_argument("--modified-dir", "--modified", dest="modified_dir", required=True)
    parser.add_argument("--suffix", default="_de-id")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resize-max-side", type=int, default=2000)
    parser.add_argument("--resize-min-side", type=int, default=161)
    parser.add_argument("--resize-small-side", type=int, default=200)
    parser.add_argument("--output", default="evaluation_results/similarity.json")
    return parser.parse_args()


def main():
    args = parse_args()
    result = calculate_similarity(
        original_dir=args.original_dir,
        modified_dir=args.modified_dir,
        suffix=args.suffix,
        workers=args.workers,
        resize_max_side=args.resize_max_side,
        resize_min_side=args.resize_min_side,
        resize_small_side=args.resize_small_side,
    )
    result["original_dir"] = args.original_dir
    result["modified_dir"] = args.modified_dir
    write_result(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
