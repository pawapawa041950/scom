"""Build a ComfyUI API-format workflow graph for the anima (separated) model.

The graph uses separated components:
  * UNETLoader        -> models/diffusion
  * VAELoader         -> models/vae
  * CLIPLoader / DualCLIPLoader -> models/te

The result is a dict keyed by stringified node ids, ready to POST to the
ComfyUI ``/prompt`` endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Common option lists surfaced in the UI. These mirror ComfyUI's built-ins.
SAMPLERS = [
    "er_sde", "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde",
    "dpmpp_3m_sde", "dpmpp_sde", "dpm_2", "dpm_2_ancestral", "lms", "heun",
    "ddim", "uni_pc",
]
SCHEDULERS = [
    "normal", "karras", "exponential", "sgm_uniform", "simple",
    "ddim_uniform", "beta",
]
# CLIP loader "type" values. Single-encoder uses CLIPLoader types; dual uses
# DualCLIPLoader types.
CLIP_TYPES_SINGLE = [
    "stable_diffusion", "sd3", "flux", "stable_cascade", "mochi",
    "ltxv", "pixart", "cosmos", "lumina2", "hidream", "chroma",
    # Qwen-Image family. krea2 = Krea-2 (Qwen3-VL text encoder).
    "qwen_image", "krea2",
]
CLIP_TYPES_DUAL = ["sdxl", "sd3", "flux", "hunyuan_video", "hidream"]


# Defaults tuned for anima (Qwen-Image based) per the reference workflow.
DEFAULT_NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, "
    "jpeg artifacts, sepia"
)


@dataclass
class GenParams:
    diffusion: str
    vae: str
    te: list[str] = field(default_factory=list)  # 1 -> CLIPLoader, 2 -> DualCLIPLoader
    clip_type: str = "stable_diffusion"
    prompt: str = ""
    negative: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg: float = 4.0
    sampler: str = "er_sde"
    scheduler: str = "simple"
    seed: int = 0
    batch_size: int = 1
    weight_dtype: str = "default"  # default | fp8_e4m3fn | fp8_e5m2
    filename_prefix: str = "scom"


def build_graph(p: GenParams) -> dict:
    """Return a ComfyUI API-format prompt graph for the given parameters."""
    if not p.diffusion:
        raise ValueError("diffusion model is required")
    if not p.vae:
        raise ValueError("vae model is required")
    if not p.te:
        raise ValueError("at least one text encoder (te) is required")

    graph: dict[str, dict] = {}

    graph["4"] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": p.diffusion, "weight_dtype": p.weight_dtype},
    }
    graph["5"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": p.vae},
    }

    if len(p.te) >= 2:
        graph["6"] = {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": p.te[0],
                "clip_name2": p.te[1],
                "type": p.clip_type,
            },
        }
    else:
        graph["6"] = {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": p.te[0], "type": p.clip_type},
        }

    graph["7"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": p.prompt, "clip": ["6", 0]},
    }
    graph["8"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": p.negative, "clip": ["6", 0]},
    }
    graph["9"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": p.width,
            "height": p.height,
            "batch_size": p.batch_size,
        },
    }
    graph["10"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": p.seed,
            "steps": p.steps,
            "cfg": p.cfg,
            "sampler_name": p.sampler,
            "scheduler": p.scheduler,
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["7", 0],
            "negative": ["8", 0],
            "latent_image": ["9", 0],
        },
    }
    graph["11"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["10", 0], "vae": ["5", 0]},
    }
    # PreviewImage writes to ComfyUI's temp dir (throwaway). The app saves the
    # real, format/quality-controlled file to output/ from the returned bytes.
    graph["12"] = {
        "class_type": "PreviewImage",
        "inputs": {"images": ["11", 0]},
    }
    return graph
