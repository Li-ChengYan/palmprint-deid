import json
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_image_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def strip_generated_suffix(stem, suffix="_de-id"):
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def pair_image_directories(original_dir, modified_dir, suffix="_de-id"):
    original_files = {path.stem: path for path in collect_image_files(original_dir)}
    modified_files = {}
    for path in collect_image_files(modified_dir):
        base_name = strip_generated_suffix(path.stem, suffix=suffix)
        modified_files.setdefault(base_name, []).append(path)

    stats = {
        "total_originals": len(original_files),
        "missing": 0,
        "multi_matches": 0,
        "total_pairs": 0,
    }
    pairs = []
    for stem, original_path in sorted(original_files.items()):
        candidates = sorted(modified_files.get(stem, []))
        if not candidates:
            stats["missing"] += 1
            continue
        if len(candidates) > 1:
            stats["multi_matches"] += 1
        for modified_path in candidates:
            pairs.append((original_path, modified_path))

    stats["total_pairs"] = len(pairs)
    return pairs, stats


def load_evaluation_config(config_path):
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("Evaluation config must contain a JSON object.")
    return config
