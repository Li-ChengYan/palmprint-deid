import argparse
import csv
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = CODE_DIR / "checkpoints" / "model.ckpt"
DEFAULT_CONFIG_PATH = CODE_DIR / "configs" / "palmprint_deid_inference.yaml"
DEFAULT_SAM_CHECKPOINT = CODE_DIR / "checkpoints" / "sam2.1_hiera_large.pt"
DEFAULT_SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SUPPORTED_MODEL_NAMES = {"ours", "palmprint-deid", "palmprint_deid"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ProfileResult:
    method: str
    model_name: str
    checkpoint: str
    input_size: tuple
    batch_size: int
    device: str
    device_name: str
    total_params_m: float
    trainable_params_m: float
    flops_g: float | None
    latency_ms_per_image: float
    average_peak_memory_mb: float | None
    max_peak_memory_mb: float | None
    flops_status: str


def parse_input_size(value):
    normalized = value.lower().replace(",", "x").replace(" ", "")
    parts = [part for part in normalized.split("x") if part]
    if len(parts) == 1:
        channels, height, width = 3, int(parts[0]), int(parts[0])
    elif len(parts) == 2:
        channels, height, width = 3, int(parts[0]), int(parts[1])
    elif len(parts) == 3:
        channels, height, width = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        raise argparse.ArgumentTypeError(
            "input size must be formatted as 512, 512x512, or 3x512x512"
        )
    if channels <= 0 or height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("input size dimensions must be positive")
    return channels, height, width


def format_input_size(input_size):
    channels, height, width = input_size
    return f"{channels} x {height} x {width}"


def format_optional_float(value, digits=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.{digits}f}"


def markdown_output_path_for(csv_path):
    return Path(csv_path).with_suffix(".md")


def collect_image_paths(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No input images found in: {image_dir}")
    return image_paths


def load_image_tensor(path, input_size):
    import numpy as np
    import torch
    from PIL import Image

    channels, height, width = input_size
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    if channels == 1:
        tensor = tensor[:1]
    elif channels != 3:
        raise ValueError("Only 1-channel or 3-channel image profiling inputs are supported.")
    return tensor


def build_preloaded_dataloader(image_dir, input_size, batch_size):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    image_tensors = [load_image_tensor(path, input_size) for path in collect_image_paths(image_dir)]
    stacked = torch.stack(image_tensors, dim=0)
    return DataLoader(TensorDataset(stacked), batch_size=batch_size, shuffle=False, num_workers=0)


def extract_input_tensor(batch):
    import torch

    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("Empty batch tuple/list received from dataloader.")
        return extract_input_tensor(batch[0])
    if isinstance(batch, dict):
        for key in ("image", "img", "input", "x"):
            if key in batch:
                return extract_input_tensor(batch[key])
        raise ValueError("Batch dict must contain one of: image, img, input, x")
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def tensor_to_bgr_uint8(image_tensor):
    import cv2
    import numpy as np

    tensor = image_tensor.detach().cpu().float()
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with shape CxHxW, got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[0]}")
    tensor = tensor.clamp(0.0, 1.0)
    rgb = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def unique_parameters(modules):
    seen = set()
    for module in modules:
        for parameter in module.parameters(recurse=True):
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter


def count_parameters(modules):
    total = 0
    trainable = 0
    for parameter in unique_parameters(modules):
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return total / 1e6, trainable / 1e6


def is_flop_profiler_hook(hook):
    hook_module = getattr(hook, "__module__", "") or ""
    hook_name = getattr(hook, "__name__", "") or ""
    hook_repr = repr(hook)
    hook_text = f"{hook_module}.{hook_name} {hook_repr}".lower()
    return "thop" in hook_text or "fvcore" in hook_text


def cleanup_flop_profiler_state(model):
    for module in model.modules():
        forward_hooks = getattr(module, "_forward_hooks", None)
        if forward_hooks is not None:
            for hook_id, hook in list(forward_hooks.items()):
                if is_flop_profiler_hook(hook):
                    del forward_hooks[hook_id]

        for attr_name in ("total_ops", "total_params"):
            if hasattr(module, attr_name):
                delattr(module, attr_name)


def collect_torch_modules_from_object(obj, max_depth=2):
    import torch

    modules = []
    seen_objects = set()
    seen_modules = set()

    def visit(current, depth):
        object_id = id(current)
        if object_id in seen_objects:
            return
        seen_objects.add(object_id)

        if isinstance(current, torch.nn.Module):
            if object_id not in seen_modules:
                seen_modules.add(object_id)
                modules.append(current)
            return

        if depth <= 0:
            return

        for attr_name in dir(current):
            if attr_name.startswith("_"):
                continue
            try:
                value = getattr(current, attr_name)
            except Exception:
                continue
            if callable(value) or isinstance(value, (str, bytes, int, float, bool, type(None))):
                continue
            visit(value, depth - 1)

    visit(obj, max_depth)
    return modules


def build_ours_model(args, device):
    import numpy as np
    import torch
    from PIL import Image

    from palm_inpainter import PalmInpainter
    from run_palm_deidentification import (
        PALM_BORDER_KEYPOINTS,
        SAM2ImagePredictor,
        WiLorHandPose3dEstimationPipeline,
        build_sam2,
        crop_with_bbox,
        extract_roi,
        get_bbox_from_points,
        get_hand_detections,
        get_segmentation_result,
        make_square_bbox,
        process_mask,
        shrink_bbox,
        to_pil_image,
    )

    class PalmprintDeidProfileWrapper(torch.nn.Module):
        expects_cpu_input = True

        def __init__(self):
            super().__init__()
            self.device = torch.device(device)
            self.reference_mode = args.reference_mode
            self.guidance_scale = args.guidance_scale
            self.interpolation_ratio = args.interpolation_ratio

            self.inpainter = PalmInpainter(
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                seed=args.seed,
                generator_seed=args.generator_seed or args.seed,
                image_height=args.input_size[1],
                image_width=args.input_size[2],
            )
            self.inpainter.device = self.device
            self.inpainter.model = self.inpainter.model.to(self.device)
            self.inpainter.start_code = self.inpainter._build_start_code(
                generator_seed=args.generator_seed or args.seed,
                height=args.input_size[1],
                width=args.input_size[2],
                downsample_factor=self.inpainter.latent_downsample_factor,
            )
            self.inpainter.conditioning_dtype = (
                torch.float16 if self.device.type == "cuda" else torch.float32
            )
            self.diffusion_model = self.inpainter.model

            try:
                sam_model = build_sam2(args.sam_config, args.sam_checkpoint, device=str(self.device))
            except TypeError:
                sam_model = build_sam2(args.sam_config, args.sam_checkpoint)
                if hasattr(sam_model, "to"):
                    sam_model = sam_model.to(self.device)
            self.sam_predictor = SAM2ImagePredictor(sam_model)
            self.sam_model = sam_model if isinstance(sam_model, torch.nn.Module) else None

            pipeline_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.keypoint_pipeline = WiLorHandPose3dEstimationPipeline(
                device=self.device,
                dtype=pipeline_dtype,
                verbose=False,
            )

            modules = [self.diffusion_model]
            if self.sam_model is not None:
                modules.append(self.sam_model)
            modules.extend(collect_torch_modules_from_object(self.keypoint_pipeline, max_depth=2))
            unique_modules = []
            seen = set()
            for module in modules:
                if id(module) not in seen:
                    seen.add(id(module))
                    unique_modules.append(module)
            self.modules_for_cost = torch.nn.ModuleList(unique_modules)

        def modules_to_profile(self):
            return list(self.modules_for_cost)

        def forward(self, batch):
            outputs = []
            for image_tensor in batch:
                image = tensor_to_bgr_uint8(image_tensor)
                detections = get_hand_detections(self.keypoint_pipeline, image)
                if not detections:
                    raise RuntimeError("No hand was detected in a profiling input image.")

                hand_keypoints = detections[0]["keypoints_2d"]
                border_points = hand_keypoints[PALM_BORDER_KEYPOINTS].astype(np.int32)

                hand_bbox = make_square_bbox(detections[0]["bbox"], image.shape[0], image.shape[1])
                roi_bbox = make_square_bbox(
                    get_bbox_from_points(border_points), image.shape[0], image.shape[1]
                )
                half_roi_bbox = shrink_bbox(roi_bbox, scale=0.5)
                detail_roi_bbox = shrink_bbox(roi_bbox, scale=0.1)

                masks, _ = get_segmentation_result(self.sam_predictor, image, hand_keypoints)
                hand_mask = process_mask(masks[0])

                roi_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
                import cv2

                roi_mask = cv2.fillConvexPoly(roi_mask, border_points, 1)
                palm_mask = roi_mask * hand_mask

                cropped_image = crop_with_bbox(image, hand_bbox)
                cropped_mask = crop_with_bbox(palm_mask, hand_bbox)

                source_image = to_pil_image(cropped_image.copy())
                global_reference_image = to_pil_image(extract_roi(image.copy(), roi_bbox))
                local_reference_image = to_pil_image(extract_roi(image.copy(), half_roi_bbox))
                detail_reference_image = to_pil_image(extract_roi(image.copy(), detail_roi_bbox))
                mask_image = Image.fromarray(
                    (cv2.resize(cropped_mask, (512, 512)) * 255).astype(np.uint8)
                )

                synthesized_palm = self.inpainter.paint_with_mask(
                    source_image=source_image,
                    global_reference_image=global_reference_image,
                    local_reference_image=local_reference_image,
                    detail_reference_image=detail_reference_image,
                    mask_image=mask_image,
                    reference_mode=self.reference_mode,
                    guidance_scale=self.guidance_scale,
                    interpolation_ratio=self.interpolation_ratio,
                )
                outputs.append(synthesized_palm)
            return outputs

    return PalmprintDeidProfileWrapper()


def build_model(args, device):
    model_name = args.model_name.lower()
    if model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model '{args.model_name}'. This repository does not contain baseline "
            "model-building code, so automatic profiling is limited to: Ours."
        )
    return build_ours_model(args, device)


def batches_from_dataloader(dataloader):
    batches = list(dataloader)
    if not batches:
        raise ValueError("Dataloader produced no batches.")
    return batches


def profile_model(model, dataloader, device="cuda", warmup=20, repeat=100):
    import torch

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    model.eval()
    model.to(torch_device)
    cleanup_flop_profiler_state(model)
    input_device = torch.device("cpu") if getattr(model, "expects_cpu_input", False) else torch_device

    batches = batches_from_dataloader(dataloader)
    latencies = []
    peak_memories = []
    measured_iterations = repeat if repeat and repeat > 0 else len(batches)

    with torch.inference_mode():
        for batch in itertools.islice(itertools.cycle(batches), warmup):
            x = extract_input_tensor(batch).to(input_device)
            _ = model(x)

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)

        for batch in itertools.islice(itertools.cycle(batches), measured_iterations):
            x = extract_input_tensor(batch).to(input_device)

            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(torch_device)
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)

                starter.record()
                _ = model(x)
                ender.record()

                torch.cuda.synchronize(torch_device)
                elapsed_ms = starter.elapsed_time(ender)
                peak_memory_mb = torch.cuda.max_memory_allocated(torch_device) / 1024 / 1024
                peak_memories.append(peak_memory_mb)
            else:
                start = time.perf_counter()
                _ = model(x)
                elapsed_ms = (time.perf_counter() - start) * 1000.0

            latencies.append(elapsed_ms / x.size(0))

    avg_latency = sum(latencies) / len(latencies)
    avg_memory = sum(peak_memories) / len(peak_memories) if peak_memories else None
    max_memory = max(peak_memories) if peak_memories else None
    return avg_latency, avg_memory, max_memory


def compute_flops_g(model, sample_batch, device):
    import torch

    input_device = torch.device("cpu") if getattr(model, "expects_cpu_input", False) else torch.device(device)
    sample = extract_input_tensor(sample_batch).to(input_device)
    cleanup_flop_profiler_state(model)

    try:
        from thop import profile as thop_profile
    except ImportError:
        thop_profile = None

    if thop_profile is not None:
        try:
            macs, _ = thop_profile(model, inputs=(sample,), verbose=False)
            return float(macs) / 1e9, "computed with thop; thop reports MACs"
        except Exception as error:
            return None, f"thop is installed but could not profile this multi-stage pipeline: {error}"
        finally:
            cleanup_flop_profiler_state(model)

    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        FlopCountAnalysis = None

    if FlopCountAnalysis is not None:
        try:
            flops = FlopCountAnalysis(model, sample).total()
            return float(flops) / 1e9, "computed with fvcore"
        except Exception as error:
            return None, f"fvcore is installed but could not profile this multi-stage pipeline: {error}"
        finally:
            cleanup_flop_profiler_state(model)

    try:
        import ptflops  # noqa: F401
    except ImportError:
        return None, "FLOPs unavailable: thop, fvcore, and ptflops are not installed."

    return None, (
        "ptflops is installed, but automatic profiling of this multi-input de-identification "
        "pipeline is not supported by this script."
    )


def get_device_name(device):
    import torch

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch_device)
    return str(torch_device)


def write_csv(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "Method": result.method,
        "Model": result.model_name,
        "Checkpoint": result.checkpoint,
        "Input size": format_input_size(result.input_size),
        "Batch size": result.batch_size,
        "Device": result.device_name,
        "Total Params (M)": format_optional_float(result.total_params_m),
        "Trainable Params (M)": format_optional_float(result.trainable_params_m),
        "FLOPs (G)": format_optional_float(result.flops_g),
        "Latency (ms/image)": format_optional_float(result.latency_ms_per_image),
        "Average Peak GPU Memory (MB)": format_optional_float(result.average_peak_memory_mb),
        "Max Peak GPU Memory (MB)": format_optional_float(result.max_peak_memory_mb),
        "FLOPs Status": result.flops_status,
    }
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_markdown(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = "\n".join(
        [
            "| Method | Params (M) ↓ | FLOPs (G) ↓ | Latency (ms/image) ↓ | Peak GPU Memory (MB) ↓ |",
            "|---|---:|---:|---:|---:|",
            (
                f"| {result.method} | {format_optional_float(result.total_params_m)} | "
                f"{format_optional_float(result.flops_g)} | "
                f"{format_optional_float(result.latency_ms_per_image)} | "
                f"{format_optional_float(result.max_peak_memory_mb)} |"
            ),
        ]
    )
    output_path.write_text(table + "\n", encoding="utf-8")


def print_summary(result):
    print("\nComputational Cost Summary")
    print(f"Model: {result.model_name}")
    print(f"Checkpoint: {result.checkpoint}")
    print(f"Input size: {format_input_size(result.input_size)}")
    print(f"Batch size: {result.batch_size}")
    print(f"Device: {result.device_name}")
    print()
    print(f"Total Params: {format_optional_float(result.total_params_m)} M")
    print(f"Trainable Params: {format_optional_float(result.trainable_params_m)} M")
    print(f"FLOPs: {format_optional_float(result.flops_g)} G")
    print(f"Latency: {format_optional_float(result.latency_ms_per_image)} ms/image")
    print(f"Average Peak GPU Memory: {format_optional_float(result.average_peak_memory_mb)} MB")
    print(f"Max Peak GPU Memory: {format_optional_float(result.max_peak_memory_mb)} MB")
    if result.flops_g is None:
        print(f"FLOPs note: {result.flops_status}")


def parse_args():
    parser = argparse.ArgumentParser(description="Profile de-identification computational cost.")
    parser.add_argument("--model", "--model-name", dest="model_name", default="ours")
    parser.add_argument("--method-name", default="Ours")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--data-root", "--image-dir", dest="image_dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--sam-checkpoint", default=str(DEFAULT_SAM_CHECKPOINT))
    parser.add_argument("--sam-config", default=DEFAULT_SAM_CONFIG)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--input-size", type=parse_input_size, default=parse_input_size("3x512x512"))
    parser.add_argument("--output", default="computational_cost.csv")
    parser.add_argument("--reference-mode", choices=["fusion", "global", "local"], default="fusion")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--interpolation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--generator-seed", type=int, default=None)
    parser.add_argument("--skip-flops", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive.")
    for path in (args.checkpoint, args.config, args.sam_checkpoint):
        if not Path(path).exists():
            raise FileNotFoundError(f"Required file does not exist: {path}")


def main():
    args = parse_args()
    validate_args(args)

    dataloader = build_preloaded_dataloader(args.image_dir, args.input_size, args.batch_size)
    model = build_model(args, args.device)
    modules = model.modules_to_profile() if hasattr(model, "modules_to_profile") else [model]
    total_params_m, trainable_params_m = count_parameters(modules)

    sample_batch = batches_from_dataloader(dataloader)[0]
    if args.skip_flops:
        flops_g = None
        flops_status = "Skipped by --skip-flops."
    else:
        flops_g, flops_status = compute_flops_g(model, sample_batch, args.device)

    latency_ms, avg_memory_mb, max_memory_mb = profile_model(
        model,
        dataloader,
        device=args.device,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    result = ProfileResult(
        method=args.method_name,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        input_size=args.input_size,
        batch_size=args.batch_size,
        device=args.device,
        device_name=get_device_name(args.device),
        total_params_m=total_params_m,
        trainable_params_m=trainable_params_m,
        flops_g=flops_g,
        latency_ms_per_image=latency_ms,
        average_peak_memory_mb=avg_memory_mb,
        max_peak_memory_mb=max_memory_mb,
        flops_status=flops_status,
    )

    csv_path = Path(args.output)
    write_csv(result, csv_path)
    write_markdown(result, markdown_output_path_for(csv_path))
    print_summary(result)


if __name__ == "__main__":
    main()
