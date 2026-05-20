import argparse
import json
from pathlib import Path

from cal_fid import calculate_fid
from cal_lpips import calculate_lpips
from cal_similarity import calculate_similarity
from evaluation_utils import load_evaluation_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "evaluation_direct_images.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Run image-quality evaluation from a JSON config file.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--original-dir", default=None)
    parser.add_argument("--modified-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def config_value(config, key, default=None):
    return config[key] if key in config else default


def main():
    args = parse_args()
    config = load_evaluation_config(args.config)

    original_dir = args.original_dir or config_value(config, "original_dir")
    modified_dir = args.modified_dir or config_value(config, "modified_dir")
    output_dir = Path(args.output_dir or config_value(config, "output_dir", "evaluation_results"))
    metrics = config_value(config, "metrics", ["similarity", "lpips", "fid"])
    suffix = config_value(config, "suffix", "_de-id")
    unknown_metrics = sorted(set(metrics) - {"similarity", "lpips", "fid"})
    if unknown_metrics:
        raise ValueError(f"Unknown evaluation metric(s): {', '.join(unknown_metrics)}")

    if not original_dir or not modified_dir:
        raise ValueError("Both original_dir and modified_dir must be set in the config or CLI arguments.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "original_dir": original_dir,
        "modified_dir": modified_dir,
        "suffix": suffix,
        "metrics": {},
    }

    if "similarity" in metrics:
        similarity = calculate_similarity(
            original_dir=original_dir,
            modified_dir=modified_dir,
            suffix=suffix,
            workers=config_value(config, "workers"),
            resize_max_side=config_value(config, "resize_max_side", 2000),
            resize_min_side=config_value(config, "resize_min_side", 161),
            resize_small_side=config_value(config, "resize_small_side", 200),
        )
        summary["metrics"]["similarity"] = similarity
        with (output_dir / "similarity.json").open("w", encoding="utf-8") as file:
            json.dump(similarity, file, indent=2)
            file.write("\n")

    if "lpips" in metrics:
        lpips_result = calculate_lpips(
            original_dir=original_dir,
            modified_dir=modified_dir,
            suffix=suffix,
            net=config_value(config, "lpips_net", "vgg"),
            resize_max_side=config_value(config, "resize_max_side", 2000),
        )
        summary["metrics"]["lpips"] = lpips_result
        with (output_dir / "lpips.json").open("w", encoding="utf-8") as file:
            json.dump(lpips_result, file, indent=2)
            file.write("\n")

    if "fid" in metrics:
        fid = calculate_fid(
            original_dir=original_dir,
            modified_dir=modified_dir,
            batch_size=config_value(config, "fid_batch_size", 50),
            dims=config_value(config, "fid_dims", 2048),
            num_workers=config_value(config, "fid_num_workers", 1),
        )
        summary["metrics"]["fid"] = {
            "metric": "FID",
            "fid": float(fid),
            "batch_size": config_value(config, "fid_batch_size", 50),
            "dims": config_value(config, "fid_dims", 2048),
        }
        with (output_dir / "fid.json").open("w", encoding="utf-8") as file:
            json.dump(summary["metrics"]["fid"], file, indent=2)
            file.write("\n")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
