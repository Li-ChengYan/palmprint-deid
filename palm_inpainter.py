from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.transforms import Resize

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config

CODE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = CODE_DIR / "checkpoints" / "model.ckpt"
DEFAULT_CONFIG_PATH = CODE_DIR / "configs" / "palmprint_deid_inference.yaml"


class PalmInpainter:
    """Palmprint inpainting wrapper built on the latent diffusion model."""

    def __init__(
        self,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        config_path=DEFAULT_CONFIG_PATH,
        seed=123,
        generator_seed=123,
        ddim_steps=50,
        guidance_scale=3.0,
        use_plms=True,
        latent_downsample_factor=8,
        image_height=512,
        image_width=512,
        latent_channels=4,
        num_samples=1,
        ddim_eta=0.0,
        precision="autocast",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.ddim_steps = ddim_steps
        self.guidance_scale = guidance_scale
        self.use_plms = use_plms
        self.latent_downsample_factor = latent_downsample_factor
        self.image_height = image_height
        self.image_width = image_width
        self.latent_channels = latent_channels
        self.num_samples = num_samples
        self.ddim_eta = ddim_eta
        self.precision = precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator_seed = generator_seed

        seed_everything(seed)
        config = OmegaConf.load(str(self.config_path))
        self.model = self._load_model_from_config(config, self.checkpoint_path)

        if self.use_plms:
            self.sampler = PLMSSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)

        self.start_code = self._build_start_code(
            generator_seed=generator_seed,
            height=self.image_height,
            width=self.image_width,
            downsample_factor=self.latent_downsample_factor,
        )
        self.conditioning_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

    def _load_model_from_config(self, config, checkpoint_path, verbose=False):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if "global_step" in checkpoint:
            print(f"Global Step: {checkpoint['global_step']}")

        model = instantiate_from_config(config.model)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if missing_keys and verbose:
            print("Missing keys:")
            print(missing_keys)
        if unexpected_keys and verbose:
            print("Unexpected keys:")
            print(unexpected_keys)

        model = model.to(self.device)
        model.eval()
        return model

    def _build_start_code(self, generator_seed, height, width, downsample_factor):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(generator_seed)
        return torch.randn(
            [
                self.num_samples,
                self.latent_channels,
                height // downsample_factor,
                width // downsample_factor,
            ],
            generator=generator,
            device=self.device,
        )

    def _get_image_transform(self, normalize=True, to_tensor=True):
        transforms = []
        if to_tensor:
            transforms.append(torchvision.transforms.ToTensor())
        if normalize:
            transforms.append(
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            )
        return torchvision.transforms.Compose(transforms)

    def _get_clip_transform(self, normalize=True, to_tensor=True):
        transforms = []
        if to_tensor:
            transforms.append(torchvision.transforms.ToTensor())
        if normalize:
            transforms.append(
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                )
            )
        return torchvision.transforms.Compose(transforms)

    def _get_autocast_context(self):
        if self.precision == "autocast" and self.device.type == "cuda":
            return autocast("cuda")
        return nullcontext()

    def paint_with_mask(
        self,
        source_image,
        global_reference_image,
        local_reference_image,
        detail_reference_image,
        mask_image,
        reference_mode="fusion",
        start_seed=None,
        guidance_scale=None,
        interpolation_ratio=0.0,
        height=None,
        width=None,
        downsample_factor=None,
    ):
        height = height or self.image_height
        width = width or self.image_width
        downsample_factor = downsample_factor or self.latent_downsample_factor

        if start_seed is not None:
            if start_seed == -1:
                generator_seed = torch.randint(0, 2**32, (1,)).item()
            else:
                generator_seed = start_seed
            start_code = self._build_start_code(generator_seed, height, width, downsample_factor)
        else:
            if (
                height == self.image_height
                and width == self.image_width
                and downsample_factor == self.latent_downsample_factor
            ):
                start_code = self.start_code
            else:
                start_code = self._build_start_code(
                    self.generator_seed,
                    height,
                    width,
                    downsample_factor,
                )

        scale = self.guidance_scale if guidance_scale is None else guidance_scale

        with torch.no_grad():
            with self._get_autocast_context():
                with self.model.ema_scope():
                    image_tensor = self._get_image_transform()(source_image).unsqueeze(0)
                    global_reference_tensor = self._get_clip_transform()(
                        global_reference_image.resize((224, 224))
                    ).unsqueeze(0)

                    mask_array = np.array(mask_image.convert("L"), dtype=np.float32)[None, None]
                    mask_array = 1 - mask_array / 255.0
                    mask_array[mask_array < 0.5] = 0
                    mask_array[mask_array >= 0.5] = 1
                    mask_tensor = torch.from_numpy(mask_array)

                    inpaint_image = image_tensor * mask_tensor
                    model_kwargs = {
                        "inpaint_mask": mask_tensor.to(self.device),
                        "inpaint_image": inpaint_image.to(self.device),
                    }

                    uc = None
                    if scale != 1.0:
                        uc = self.model.learnable_vector

                    conditioning = self.model.get_learned_conditioning(
                        global_reference_tensor.to(self.device, dtype=self.conditioning_dtype)
                    )
                    conditioning = self.model.proj_out(conditioning)

                    local_reference_tensor = self._get_clip_transform()(
                        local_reference_image.resize((224, 224))
                    ).unsqueeze(0)
                    local_conditioning = self.model.get_learned_conditioning(
                        local_reference_tensor.to(self.device, dtype=self.conditioning_dtype)
                    )
                    local_conditioning = self.model.proj_out(local_conditioning)

                    detail_reference_tensor = self._get_clip_transform()(
                        detail_reference_image.resize((224, 224))
                    ).unsqueeze(0)
                    detail_conditioning = self.model.get_learned_conditioning(
                        detail_reference_tensor.to(self.device, dtype=self.conditioning_dtype)
                    )
                    detail_conditioning = self.model.proj_out(detail_conditioning)

                    if reference_mode == "fusion":
                        conditioning = (local_conditioning + detail_conditioning) / 2
                    elif reference_mode == "global":
                        conditioning = conditioning
                    elif reference_mode == "local":
                        conditioning = local_conditioning
                    else:
                        raise ValueError("reference_mode must be one of: fusion, global, local")

                    encoded_inpaint = self.model.encode_first_stage(model_kwargs["inpaint_image"])
                    encoded_inpaint = self.model.get_first_stage_encoding(encoded_inpaint).detach()

                    encoded_original = self.model.encode_first_stage(image_tensor.to(self.device))
                    encoded_original = self.model.get_first_stage_encoding(encoded_original).detach()

                    model_kwargs["inpaint_image"] = (
                        encoded_original * interpolation_ratio
                        + encoded_inpaint * (1 - interpolation_ratio)
                    )
                    model_kwargs["inpaint_mask"] = Resize(
                        [encoded_inpaint.shape[-2], encoded_inpaint.shape[-1]]
                    )(model_kwargs["inpaint_mask"])

                    latent_shape = [
                        self.latent_channels,
                        height // downsample_factor,
                        width // downsample_factor,
                    ]
                    samples, _ = self.sampler.sample(
                        S=self.ddim_steps,
                        conditioning=conditioning,
                        batch_size=self.num_samples,
                        shape=latent_shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=self.ddim_eta,
                        x_T=start_code,
                        test_model_kwargs=model_kwargs,
                    )

                    decoded_samples = self.model.decode_first_stage(samples)
                    decoded_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0) * 255
                    return (
                        decoded_samples.squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                    )
