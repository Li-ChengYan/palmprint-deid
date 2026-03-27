# Palmprint-DeID

Official open-source implementation of **"Palmprint De-Identification Using Diffusion Model for High-Quality and Diverse Synthesis"**.

## Paper

- **Title:** Palmprint De-Identification Using Diffusion Model for High-Quality and Diverse Synthesis
- **Authors:** Licheng Yan, Bob Zhang, Andrew Beng Jin Teoh, Lu Leng, Shuyi Li, Yuqi Wang, Ziyuan Yang
- **Status:** arXiv preprint, 2025
- **Paper link:** [arXiv:2504.08272](https://arxiv.org/abs/2504.08272)

## Overview

This repository contains the official code release for our palmprint de-identification pipeline. The method is built on a diffusion inpainting framework and improves palmprint privacy protection through semantic-guided embedding fusion and prior interpolation, aiming at high-quality and diverse synthesized outputs.

This README treats the current directory as the main project directory for the released code.

By default, the code resolves `checkpoints/` and `configs/` relative to this directory, so you do not need to launch the script from a specific parent folder.

## Project Structure

```text
palm_inpainter.py
run_palm_deidentification.py
README.md
checkpoints/
  model.ckpt
  sam2.1_hiera_large.pt
configs/
  palmprint_deid_inference.yaml
```

## Environment Requirements

- Python 3.10 is recommended.
- A CUDA-capable GPU is recommended for practical inference.
- `torch>=2.5.1` and `torchvision>=0.20.1` are required by SAM 2.
- Linux is recommended. If you are using Windows, the SAM 2 team strongly recommends WSL with Ubuntu.

## Installation

Create a fresh Python 3.10 environment first:

```bash
conda create -n palmprint-deid python=3.10
conda activate palmprint-deid
```

Then install PyTorch and TorchVision following the official PyTorch instructions for your CUDA version:

```bash
# See https://pytorch.org/get-started/locally/ for the exact command.
```

After that, install the upstream dependencies required by this project:

1. Prepare Paint-by-Example so that the `ldm` modules are available:

```bash
git clone https://github.com/Fantasy-Studio/Paint-by-Example.git
cd Paint-by-Example
```

2. Install SAM 2:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

3. Install WiLoR-mini:

```bash
pip install git+https://github.com/warmshao/WiLoR-mini
```

Notes:
- The official SAM 2 repository requires `python>=3.10`, `torch>=2.5.1`, and `torchvision>=0.20.1`.
- WiLoR-mini also recommends Python 3.10.
- This project reuses the `ldm` modules from Paint-by-Example, so the Paint-by-Example source tree should remain available in your environment.

## Model Preparation

Create the directory `checkpoints/` before downloading the checkpoints.

Then prepare the following files.

### 1. Paint-by-Example checkpoint

Download the official `model.ckpt` released by the Paint-by-Example authors:

- Hugging Face: [model.ckpt](https://huggingface.co/Fantasy-Studio/Paint-by-Example/resolve/main/model.ckpt)
- Google Drive: [official shared checkpoint](https://drive.google.com/file/d/15QzaTWsvZonJcXsNv-ilMRCYaQLhzR_i/view?usp=share_link)

Save it as:

```text
checkpoints/model.ckpt
```

### 2. SAM 2 checkpoint

Download the official SAM 2.1 large checkpoint:

- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

Save it as:

```text
checkpoints/sam2.1_hiera_large.pt
```

## Inference

After preparing the environment and checkpoints, run:

Minimal command with default parameters:

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
  --interpolation-ratio 0.1
```

Example input directory:

```text
input_images/
  sample_001.jpg
  sample_002.png
  sample_003.bmp
```

Example output directory:

```text
output_images/
  sample_001_de-id.png
  sample_002_de-id.png
  sample_003_de-id.png
```

Main arguments:
- `--input-dir`: directory containing the palmprint images to process
- `--output-dir`: directory for the de-identified results
- `--checkpoint-path`: path to the Paint-by-Example `model.ckpt`
- `--config-path`: path to this repository's inference config
- `--sam-checkpoint`: path to `sam2.1_hiera_large.pt`
- `--reference-mode`: one of `fusion`, `global`, or `local`
- `--interpolation-ratio`: interpolation weight used in latent blending

## Acknowledgements

This project builds on several excellent open-source projects. We sincerely thank their authors and maintainers:

- [Paint-by-Example](https://github.com/Fantasy-Studio/Paint-by-Example)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [WiLoR](https://github.com/rolpotamias/WiLoR)
- [WiLoR-mini](https://github.com/warmshao/WiLoR-mini)

## Citation

If you find this project useful, please consider citing our paper:

```bibtex
@article{yan2025palmprintdeid,
  title={Palmprint De-Identification Using Diffusion Model for High-Quality and Diverse Synthesis},
  author={Yan, Licheng and Zhang, Bob and Teoh, Andrew Beng Jin and Leng, Lu and Li, Shuyi and Wang, Yuqi and Yang, Ziyuan},
  journal={arXiv preprint arXiv:2504.08272},
  year={2025}
}
```
