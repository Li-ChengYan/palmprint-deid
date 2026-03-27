import argparse
import glob
import os
import random
import warnings
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)

from palm_inpainter import PalmInpainter

warnings.filterwarnings("ignore", category=DeprecationWarning)

CODE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = CODE_DIR / "checkpoints" / "model.ckpt"
DEFAULT_CONFIG_PATH = CODE_DIR / "configs" / "palmprint_deid_inference.yaml"
DEFAULT_SAM_CHECKPOINT = CODE_DIR / "checkpoints" / "sam2.1_hiera_large.pt"
DEFAULT_SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
PALM_BORDER_KEYPOINTS = [0, 17, 13, 9, 5, 2]


def parse_args():
    parser = argparse.ArgumentParser(description="Run palmprint de-identification on a folder of images.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input images.")
    parser.add_argument("--output-dir", required=True, help="Directory to store de-identified outputs.")
    parser.add_argument(
        "--checkpoint-path",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to the diffusion model checkpoint.",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the diffusion model config file.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=str(DEFAULT_SAM_CHECKPOINT),
        help="Path to the SAM2 checkpoint.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=["fusion", "global", "local"],
        default="fusion",
        help="Conditioning mode used for inpainting.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override the diffusion guidance scale.",
    )
    parser.add_argument(
        "--interpolation-ratio",
        type=float,
        default=0.1,
        help="Interpolation ratio between the original and masked latent features.",
    )
    parser.add_argument(
        "--sam-config",
        default=DEFAULT_SAM_CONFIG,
        help="SAM2 config name or path. The default uses the config packaged with SAM 2.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Global random seed.")
    return parser.parse_args()


def validate_required_paths(paths):
    missing_paths = [str(path) for path in paths if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(
            "The following required files were not found:\n- " + "\n- ".join(missing_paths)
        )


def collect_images(images_path):
    patterns = ("*.[jp][pn]g", "*.[JP][PN]G", "*.jpeg", "*.bmp")
    images = []
    for pattern in patterns:
        images.extend(glob.glob(os.path.join(images_path, pattern)))
    if not images:
        print(f"No images found in {images_path}")
    return sorted(images)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def to_pil_image(image):
    resized_image = cv2.resize(image, (512, 512))
    return Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))


def get_hand_detections(pipeline, image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    outputs = pipeline.predict(rgb_image)

    detections = []
    for output in outputs:
        detections.append(
            {
                "keypoints_2d": output["wilor_preds"]["pred_keypoints_2d"][0],
                "bbox": output["hand_bbox"],
            }
        )
    return detections


def get_segmentation_result(predictor, image, keypoints):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        predictor.set_image(rgb_image)
        labels = np.ones(len(keypoints), dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=keypoints,
            point_labels=labels,
            multimask_output=False,
        )
    return masks, scores


def process_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    return mask


def get_bbox_from_points(keypoints):
    x_min = np.min(keypoints[:, 0])
    x_max = np.max(keypoints[:, 0])
    y_min = np.min(keypoints[:, 1])
    y_max = np.max(keypoints[:, 1])
    return np.array([x_min, y_min, x_max, y_max], dtype="float32")


def shrink_bbox(bbox, scale=0.5):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    x_min = x_min + width * (1 - scale) / 2
    x_max = x_max - width * (1 - scale) / 2
    y_min = y_min + height * (1 - scale) / 2
    y_max = y_max - height * (1 - scale) / 2
    return np.array([x_min, y_min, x_max, y_max], dtype="float32")


def bbox_to_points(bbox):
    x_min, y_min, x_max, y_max = bbox
    return np.array(
        [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]],
        dtype="float32",
    )


def extract_roi(image, bbox, side=128):
    source = bbox_to_points(bbox)
    destination = np.array([[0, 0], [side, 0], [0, side], [side, side]], dtype="float32")
    perspective_matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, perspective_matrix, (side, side))


def make_square_bbox(bbox, image_height, image_width):
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    side_length = min(max(width, height), image_width, image_height)

    new_x_min = center_x - side_length / 2.0
    new_x_max = center_x + side_length / 2.0
    new_y_min = center_y - side_length / 2.0
    new_y_max = center_y + side_length / 2.0

    if new_x_min < 0:
        new_x_min = 0.0
        new_x_max = side_length
    elif new_x_max > image_width:
        new_x_max = float(image_width)
        new_x_min = new_x_max - side_length

    if new_y_min < 0:
        new_y_min = 0.0
        new_y_max = side_length
    elif new_y_max > image_height:
        new_y_max = float(image_height)
        new_y_min = new_y_max - side_length

    return np.array([new_x_min, new_y_min, new_x_max, new_y_max], dtype="float32")


def crop_with_bbox(image, bbox):
    return image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]


def process_single_image(
    image_path,
    output_dir,
    inpainter,
    sam_predictor,
    keypoint_pipeline,
    reference_mode,
    guidance_scale,
    interpolation_ratio,
):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    detections = get_hand_detections(keypoint_pipeline, image)
    if not detections:
        raise ValueError("No hand was detected in the input image.")

    hand_keypoints = detections[0]["keypoints_2d"]
    border_points = hand_keypoints[PALM_BORDER_KEYPOINTS].astype(np.int32)

    hand_bbox = make_square_bbox(detections[0]["bbox"], image.shape[0], image.shape[1])
    roi_bbox = make_square_bbox(get_bbox_from_points(border_points), image.shape[0], image.shape[1])
    half_roi_bbox = shrink_bbox(roi_bbox, scale=0.5)
    detail_roi_bbox = shrink_bbox(roi_bbox, scale=0.1)

    masks, _ = get_segmentation_result(sam_predictor, image, hand_keypoints)
    hand_mask = process_mask(masks[0])

    roi_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    roi_mask = cv2.fillConvexPoly(roi_mask, border_points, 1)
    palm_mask = roi_mask * hand_mask

    cropped_image = crop_with_bbox(image, hand_bbox)
    cropped_mask = crop_with_bbox(palm_mask, hand_bbox)

    source_image = to_pil_image(cropped_image.copy())
    global_reference_image = to_pil_image(extract_roi(image.copy(), roi_bbox))
    local_reference_image = to_pil_image(extract_roi(image.copy(), half_roi_bbox))
    detail_reference_image = to_pil_image(extract_roi(image.copy(), detail_roi_bbox))
    mask_image = Image.fromarray((cv2.resize(cropped_mask, (512, 512)) * 255).astype(np.uint8))

    with warnings.catch_warnings():
        synthesized_palm = inpainter.paint_with_mask(
            source_image=source_image,
            global_reference_image=global_reference_image,
            local_reference_image=local_reference_image,
            detail_reference_image=detail_reference_image,
            mask_image=mask_image,
            reference_mode=reference_mode,
            guidance_scale=guidance_scale,
            interpolation_ratio=interpolation_ratio,
        )

    synthesized_palm = cv2.cvtColor(synthesized_palm, cv2.COLOR_RGB2BGR)
    cropped_height, cropped_width = cropped_image.shape[:2]
    synthesized_palm = cv2.resize(synthesized_palm, (cropped_width, cropped_height))

    recovered_image = image.copy()
    recovered_image[int(hand_bbox[1]) : int(hand_bbox[3]), int(hand_bbox[0]) : int(hand_bbox[2])] = (
        synthesized_palm
    )

    hand_mask_3d = hand_mask[:, :, None].astype(np.uint8)
    blended_image = recovered_image * hand_mask_3d + image.copy() * (1 - hand_mask_3d)
    blended_image = blended_image.astype(np.uint8)

    output_path = Path(output_dir) / f"{Path(image_path).stem}_de-id.png"
    cv2.imwrite(str(output_path), blended_image)


def run_deidentification(args):
    setup_seed(args.seed)
    validate_required_paths(
        [
            args.input_dir,
            args.checkpoint_path,
            args.config_path,
            args.sam_checkpoint,
        ]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inpainter = PalmInpainter(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        seed=args.seed,
    )

    sam_predictor = SAM2ImagePredictor(build_sam2(args.sam_config, args.sam_checkpoint))

    pipeline_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    keypoint_pipeline = WiLorHandPose3dEstimationPipeline(
        device=pipeline_device,
        dtype=pipeline_dtype,
        verbose=False,
    )

    for image_path in collect_images(args.input_dir):
        try:
            process_single_image(
                image_path=image_path,
                output_dir=output_dir,
                inpainter=inpainter,
                sam_predictor=sam_predictor,
                keypoint_pipeline=keypoint_pipeline,
                reference_mode=args.reference_mode,
                guidance_scale=args.guidance_scale,
                interpolation_ratio=args.interpolation_ratio,
            )
        except Exception as error:
            print(error)
            print(f"Failed to process {image_path}")


if __name__ == "__main__":
    run_deidentification(parse_args())
