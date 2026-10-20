from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _dtype_kwarg(dtype) -> dict:
    import inspect
    from transformers import PreTrainedModel

    try:
        params = inspect.signature(PreTrainedModel.from_pretrained).parameters
        if "dtype" in params:
            return {"dtype": dtype}
    except (ValueError, TypeError):
        pass
    return {"torch_dtype": dtype}


def is_internvl_chat(model_name: str) -> bool:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if getattr(cfg, "model_type", "") == "internvl_chat":
            return True
        archs = getattr(cfg, "architectures", None) or []
        return any(("InternVLChatModel" in a for a in archs))
    except Exception:
        return False


def _build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _closest_aspect_ratio(ar: float, targets, w: int, h: int, size: int):
    best_diff, best = (float("inf"), (1, 1))
    area = w * h
    for r in targets:
        tar = r[0] / r[1]
        diff = abs(ar - tar)
        if diff < best_diff:
            best_diff, best = (diff, r)
        elif diff == best_diff and area > 0.5 * size * size * r[0] * r[1]:
            best = r
    return best


def dynamic_preprocess(
    image, min_num: int = 1, max_num: int = 6, image_size: int = 448, use_thumbnail: bool = True
):
    w, h = image.size
    ar = w / h
    targets = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    ratio = _closest_aspect_ratio(ar, targets, w, h, image_size)
    tw, th = (image_size * ratio[0], image_size * ratio[1])
    blocks = ratio[0] * ratio[1]
    resized = image.resize((tw, th))
    tiles = []
    cols = tw // image_size
    for i in range(blocks):
        box = (
            i % cols * image_size,
            i // cols * image_size,
            (i % cols + 1) * image_size,
            (i // cols + 1) * image_size,
        )
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_pixel_values(image_paths: Sequence[str], max_num: int = 6, image_size: int = 448):
    import torch
    from PIL import Image

    tf = _build_transform(image_size)
    tiles = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        tiles.extend(dynamic_preprocess(img, max_num=max_num, image_size=image_size))
    if not tiles:
        return (None, 0)
    pv = torch.stack([tf(t) for t in tiles])
    return (pv, pv.shape[0])


def _set_sdpa(model) -> None:
    import torch

    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return
    seen = set()
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        if getattr(cfg, "_attn_implementation", None) == "eager":
            try:
                cfg._attn_implementation = "sdpa"
            except (AttributeError, TypeError):
                pass
    top = getattr(model, "config", None)
    for sub in ("llm_config", "text_config"):
        c = getattr(top, sub, None)
        if c is not None and getattr(c, "_attn_implementation", None) != "sdpa":
            try:
                c._attn_implementation = "sdpa"
            except (AttributeError, TypeError):
                pass
    logger.info("[InternVL] attention set to sdpa where it was eager")


class InternVLChatBackend:
    def __init__(
        self,
        model_name: str,
        device_map: Any,
        trainable: bool,
        lora_cfg: Optional[Dict[str, Any]] = None,
        adapter_path: Optional[str] = None,
        max_image_tiles: int = 6,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_image_tiles = max_image_tiles
        self._is_peft = False
        logger.info("[InternVL] loading %s (remote code)", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        model = AutoModel.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **_dtype_kwarg(torch.bfloat16),
        )
        model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        _set_sdpa(model)
        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
            self._is_peft = True
        elif trainable:
            from peft import LoraConfig, TaskType, get_peft_model

            lc = lora_cfg or {}
            model = get_peft_model(
                model,
                LoraConfig(
                    r=lc.get("r", 16),
                    lora_alpha=lc.get("lora_alpha", 32),
                    target_modules=lc.get("target_modules", ["q_proj", "v_proj"]),
                    lora_dropout=lc.get("lora_dropout", 0.05),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                ),
            )
            self._is_peft = True
            tr = sum((p.numel() for p in model.parameters() if p.requires_grad))
            tot = sum((p.numel() for p in model.parameters()))
            logger.info(
                "[InternVL] LoRA: %d / %d trainable (%.4f%%)", tr, tot, 100 * tr / max(tot, 1)
            )
        self.model = model
        self.model.eval()
        self.input_device = next(model.parameters()).device
        off = {}
        for _n, _p in model.named_parameters():
            if _p.device.type != "cuda":
                off[_p.device.type] = off.get(_p.device.type, 0) + 1
        if off:
            raise RuntimeError(
                f"{model_name} was partly offloaded ({off}); the GPU did not have room. Free the previous model before loading this one — an offloaded 38B runs ~100x slower and looks like a hang."
            )
        logger.info("[InternVL] ready; input device %s", self.input_device)

    def _core(self):
        m = self.model
        for attr in ("base_model", "model"):
            if hasattr(m, "batch_chat"):
                break
            m = getattr(m, attr, m)
        return m

    def language_model(self):
        core = self._core()
        lm = getattr(core, "language_model", None)
        if lm is None:
            raise AttributeError("InternVLChatModel has no language_model")
        return lm

    def generate(
        self,
        prompts: Sequence[str],
        image_paths_per_prompt: Sequence[Sequence[str]],
        n_samples: int,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> List[List[str]]:
        import torch

        core = self._core()
        self.model.eval()
        gen_cfg = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_cfg.update(temperature=temperature, top_p=top_p)
        flat_q: List[str] = []
        flat_img: List[Sequence[str]] = []
        for p, imgs in zip(prompts, image_paths_per_prompt):
            for _ in range(n_samples):
                flat_q.append(p)
                flat_img.append(imgs)
        pvs, npatch = ([], [])
        for imgs in flat_img:
            pv, n = load_pixel_values(imgs, max_num=self.max_image_tiles) if imgs else (None, 0)
            pvs.append(pv)
            npatch.append(n)
        has_img = any((n > 0 for n in npatch))
        if has_img:
            outs = [""] * len(flat_q)
            idx_img = [i for i, n in enumerate(npatch) if n > 0]
            idx_txt = [i for i, n in enumerate(npatch) if n == 0]
            if idx_img:
                pixel_values = torch.cat([pvs[i] for i in idx_img]).to(
                    self.input_device, dtype=torch.bfloat16
                )
                res = core.batch_chat(
                    self.tokenizer,
                    pixel_values,
                    questions=[f"<image>\n{flat_q[i]}" for i in idx_img],
                    generation_config=gen_cfg,
                    num_patches_list=[npatch[i] for i in idx_img],
                )
                for i, r in zip(idx_img, res):
                    outs[i] = r
            if idx_txt:
                res = core.batch_chat(
                    self.tokenizer,
                    None,
                    questions=[flat_q[i] for i in idx_txt],
                    generation_config=gen_cfg,
                    num_patches_list=[0] * len(idx_txt),
                )
                for i, r in zip(idx_txt, res):
                    outs[i] = r
        else:
            outs = core.batch_chat(
                self.tokenizer,
                None,
                questions=list(flat_q),
                generation_config=gen_cfg,
                num_patches_list=[0] * len(flat_q),
            )
        grouped: List[List[str]] = []
        for i in range(len(prompts)):
            grouped.append([(outs[i * n_samples + j] or "").strip() for j in range(n_samples)])
        return grouped

    def lm_forward(self, input_ids, attention_mask, labels):
        return self.language_model()(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def disable_adapter(self):
        return self.model.disable_adapter()

    def save_adapter(self, path: str) -> Optional[str]:
        if not self._is_peft:
            return None
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        return path
