# Palmprint-DeID

Official open-source implementation of **"Palmprint De-Identification Using Diffusion Model for High-Quality and Diverse Synthesis"**.

## Paper

- **Title:** Palmprint De-Identification Using Diffusion Model for High-Quality and Diverse Synthesis
- **Authors:** Licheng Yan, Bob Zhang, Andrew Beng Jin Teoh, Lu Leng, Shuyi Li, Yuqi Wang, Ziyuan Yang
- **Status:** arXiv preprint, 2025
- **Paper link:** [arXiv:2504.08272](https://arxiv.org/abs/2504.08272)

## Overview

This repository contains the official code release for the palmprint de-identification pipeline. The method uses a diffusion inpainting framework with semantic-guided embedding fusion and prior interpolation to synthesize high-quality de-identified palmprint images.

The code resolves `checkpoints/` and `configs/` relative to this project directory.

## Project Structure

```text
palm_inpainter.py
run_palm_deidentification.py
run_evaluation.py
cal_fid.py
cal_lpips.py
cal_similarity.py
evaluation_utils.py
README.md
checkpoints/
  model.ckpt
  sam2.1_hiera_large.pt
configs/
  palmprint_deid_inference.yaml
  evaluation_direct_images.json
```

## Environment

The environment follows the union of the upstream SAM 2 and Paint-by-Example requirements, using the newer PyTorch stack required by SAM 2.

| Item | Version / specification |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.5.1 |
| TorchVision | 0.20.1 |
| GPU used in experiments | NVIDIA GeForce RTX 4090 |

Install SAM 2, Paint-by-Example, and WiLoR-mini using their upstream instructions. SAM 2 requires the newer PyTorch stack above; Paint-by-Example provides the `ldm` modules and CLIP conditioning implementation used by this release.

## Installation

Create a fresh Python 3.10 environment:

```bash
conda create -n palmprint-deid python=3.10
conda activate palmprint-deid
```

Install PyTorch and TorchVision:

```bash
pip install torch==2.5.1 torchvision==0.20.1
```

Prepare the upstream dependencies:

```bash
git clone https://github.com/Fantasy-Studio/Paint-by-Example.git
cd Paint-by-Example
```

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

```bash
pip install git+https://github.com/warmshao/WiLoR-mini
```

Keep the Paint-by-Example source tree available so that the `ldm` modules can be imported.

## Pretrained Models

Create `checkpoints/` and place the following official checkpoints in it.

| Component | Source | Local path |
| --- | --- | --- |
| Diffusion inpainting model | Paint-by-Example official `model.ckpt` from [Hugging Face](https://huggingface.co/Fantasy-Studio/Paint-by-Example/resolve/main/model.ckpt) or the authors' [Google Drive](https://drive.google.com/file/d/15QzaTWsvZonJcXsNv-ilMRCYaQLhzR_i/view?usp=share_link) | `checkpoints/model.ckpt` |
| CLIP image encoder | Paint-by-Example internal `FrozenCLIPImageEmbedder` conditioning branch, configured in `configs/palmprint_deid_inference.yaml` | included through Paint-by-Example / `model.ckpt` |
| SAM 2 segmentation model | Official SAM 2.1 Hiera-L checkpoint [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) | `checkpoints/sam2.1_hiera_large.pt` |
| Hand keypoint detector | [WiLoR-mini](https://github.com/warmshao/WiLoR-mini) package used by `run_palm_deidentification.py` | installed as a Python package |

## Inference

Minimal command:

```bash
python run_palm_deidentification.py \
  --input-dir /path/to/input_images \
  --output-dir /path/to/output_images
```

Full command:

```bash
python run_palm_deidentification.py \
  --input-dir /path/to/input_images \
  --output-dir /path/to/output_images \
  --checkpoint-path checkpoints/model.ckpt \
  --config-path configs/palmprint_deid_inference.yaml \
  --sam-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --reference-mode fusion \
  --interpolation-ratio 0.1 \
  --seed 123 \
  --generator-seed 123
```

Input directory:

```text
input_images/
  sample_001.jpg
  sample_002.png
  sample_003.bmp
```

Output directory:

```text
output_images/
  sample_001_de-id.png
  sample_002_de-id.png
  sample_003_de-id.png
```

Main arguments:

- `--input-dir`: directory containing palmprint images.
- `--output-dir`: directory for de-identified results.
- `--checkpoint-path`: Paint-by-Example diffusion checkpoint.
- `--config-path`: diffusion inference config.
- `--sam-checkpoint`: SAM 2.1 Hiera-L checkpoint.
- `--sam-config`: SAM 2 config name or path. The default uses the config packaged with SAM 2.
- `--reference-mode`: one of `fusion`, `global`, or `local`.
- `--interpolation-ratio`: interpolation weight used in latent blending.
- `--seed`: global random seed. Default: `123`.
- `--generator-seed`: diffusion initial-noise seed. If omitted, it defaults to `--seed`.

Random seed handling: `--seed` is applied to Python `random`, NumPy, PyTorch CPU/CUDA RNGs, `PYTHONHASHSEED`, and CuDNN deterministic flags. `--generator-seed` controls the latent diffusion start code used by `PalmInpainter`.

## Evaluation

The released evaluation protocol directly compares two image folders:

- `original_dir`: original palmprint images.
- `modified_dir`: de-identified images generated by this method or a baseline.
- Default filename rule: `sample.png` in `original_dir` matches `sample_de-id.png` in `modified_dir`.

Run all released image-quality metrics with the config file:

```bash
python run_evaluation.py \
  --config configs/evaluation_direct_images.json \
  --original-dir /path/to/original_images \
  --modified-dir /path/to/deidentified_images \
  --output-dir evaluation_results
```

The config file [configs/evaluation_direct_images.json](configs/evaluation_direct_images.json) records the default direct-image evaluation settings. The unified runner writes:

```text
evaluation_results/
  summary.json
  similarity.json
  lpips.json
  fid.json
```

Individual metrics can also be run directly:

```bash
python cal_similarity.py --original-dir /path/to/original_images --modified-dir /path/to/deidentified_images
python cal_lpips.py --original-dir /path/to/original_images --modified-dir /path/to/deidentified_images
python cal_fid.py --original-dir /path/to/original_images --modified-dir /path/to/deidentified_images
```

Released metrics:

- FID
- LPIPS
- PSNR
- SSIM
- MS-SSIM

## Acknowledgements

This project builds on several open-source projects:

- [Paint-by-Example](https://github.com/Fantasy-Studio/Paint-by-Example)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [WiLoR](https://github.com/rolpotamias/WiLoR)
- [WiLoR-mini](https://github.com/warmshao/WiLoR-mini)

## Citation

If you find this project useful, please cite:

```bibtex
@article{yan2025palmprintdeid,
  title={Palmprint De-Identification Using Diffusion Model for High-Quality and Diverse Synthesis},
  author={Yan, Licheng and Zhang, Bob and Teoh, Andrew Beng Jin and Leng, Lu and Li, Shuyi and Wang, Yuqi and Yang, Ziyuan},
  journal={arXiv preprint arXiv:2504.08272},
  year={2025}
}
```
