"""Build a ComfyUI API-format workflow graph for the anima (separated) model.

The graph uses separated components:
  * UNETLoader        -> models/diffusion
  * VAELoader         -> models/vae
  * CLIPLoader / DualCLIPLoader -> models/te

The result is a dict keyed by stringified node ids, ready to POST to the
ComfyUI ``/prompt`` endpoint.
"""
from __future__ import annotations

import json
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
    # Optional model merge: [(filename, weight), ...]. When non-empty the
    # diffusion model comes from the ScomMergeModel custom node (weighted
    # average materialized in RAM; see app/comfy_custom_nodes.py) and
    # ``diffusion`` is ignored. Weights are relative (need not sum to 1).
    # A single entry is just that model, optionally quantized, pinned in RAM.
    merge_models: list[tuple[str, float]] = field(default_factory=list)
    # Output precision of the merge: "" (bf16) | "fp8" | "int8_convrot".
    merge_quant: str = ""
    # True: fold sources one at a time (low RAM); False: all at once (fp32).
    merge_low_memory: bool = False
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


# Quantization choices for the merged model (node input "quantize").
MERGE_QUANT_MODES = ("", "fp8", "int8_convrot")


def _validate_merge(merge_models: list[tuple[str, float]],
                    quant: str = "") -> None:
    # A single model is allowed: the "merge" is then just (optionally
    # quantized) materialization of that model into backend RAM.
    if len(merge_models) < 1:
        raise ValueError("マージには1個以上のモデルが必要です")
    for name, w in merge_models:
        if not name:
            raise ValueError("マージ対象のモデル名が空です")
        if w <= 0:
            raise ValueError(f"マージ比率は正の数値が必要です: {name} = {w}")
    if quant not in MERGE_QUANT_MODES:
        raise ValueError(f"不明な量子化形式です: {quant}")


def merge_recipe(merge_models: list[tuple[str, float]]) -> str:
    """The merge node's recipe input. A stable string matters: both ComfyUI's
    output cache and the node's own pin cache key on it, so an identical
    config reuses the merged model already sitting in RAM."""
    return json.dumps([[n, float(w)] for n, w in merge_models])


def merge_pin_key(merge_models: list[tuple[str, float]], quant: str,
                  low_memory: bool) -> str:
    """Key of the backend pin cache entry (must mirror the node's _pin_key)."""
    return json.dumps([merge_recipe(merge_models), quant, bool(low_memory)])


def _merge_node(merge_models: list[tuple[str, float]], quant: str,
                low_memory: bool, save_to: str = "") -> dict:
    return {
        "class_type": "ScomMergeModel",
        "inputs": {"recipe": merge_recipe(merge_models), "quantize": quant,
                   "low_memory": bool(low_memory), "save_to": save_to},
    }


def build_merge_graph(merge_models: list[tuple[str, float]], quant: str = "",
                      low_memory: bool = False, save_to: str = "") -> dict:
    """Merge-only prompt: build (or refresh) the merged model in backend RAM.

    With ``save_to`` the merged model is also written to the diffusion_models
    folder as a safetensors file. The node id matches build_graph's diffusion
    node, so a following generation with the same config is a cache hit.
    """
    _validate_merge(merge_models, quant)
    return {"4": _merge_node(merge_models, quant, low_memory, save_to)}


def build_graph(p: GenParams) -> dict:
    """Return a ComfyUI API-format prompt graph for the given parameters."""
    merging = len(p.merge_models) >= 1
    if not merging and not p.diffusion:
        raise ValueError("diffusion model is required")
    if not p.vae:
        raise ValueError("vae model is required")
    if not p.te:
        raise ValueError("at least one text encoder (te) is required")

    graph: dict[str, dict] = {}

    if merging:
        _validate_merge(p.merge_models, p.merge_quant)
        graph["4"] = _merge_node(p.merge_models, p.merge_quant,
                                 p.merge_low_memory)
    else:
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
