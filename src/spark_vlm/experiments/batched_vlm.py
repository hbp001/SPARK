from __future__ import annotations
import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _valid_image(p: Path) -> bool:
    try:
        if p.stat().st_size < 100:
            return False
        from PIL import Image

        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def _split_by_images(requests, text_bs: int, img_bs: int, max_images: int):
    out, cur, cur_has_img = ([], [], None)
    for r in requests:
        has = bool(
            [
                p
                for p in list(r.image_paths)[:max_images]
                if Path(p).exists() and _valid_image(Path(p))
            ]
        )
        limit = img_bs if has else text_bs
        if cur and (has != cur_has_img or len(cur) >= limit):
            out.append(cur)
            cur = []
        cur.append(r)
        cur_has_img = has
    if cur:
        out.append(cur)
    return out


@dataclass
class GenRequest:
    text: str
    image_paths: Sequence[str] = ()
    tag: str = ""


class BatchedVLM:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        lora_cfg: Optional[Dict[str, Any]] = None,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 384,
        temperature: float = 0.8,
        top_p: float = 0.9,
        gen_batch_size: int = 32,
        max_images_per_prompt: int = 2,
        max_prompt_tokens: int = 1024,
        trainable: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.lora_cfg = lora_cfg or {}
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.gen_batch_size = gen_batch_size
        self.max_images_per_prompt = max_images_per_prompt
        self.max_prompt_tokens = max_prompt_tokens
        self.trainable = trainable
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.backend = None
        self.input_device = device
        self._is_peft = False
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor
        from spark_vlm.experiments.internvl_backend import InternVLChatBackend, is_internvl_chat

        device_map_early: Any = "auto" if self.device in ("auto", "multi") else self.device
        if is_internvl_chat(self.model_name):
            self.backend = InternVLChatBackend(
                self.model_name, device_map_early, self.trainable, self.lora_cfg, self.adapter_path
            )
            self.model = self.backend.model
            self.tokenizer = self.backend.tokenizer
            self.processor = None
            self._is_peft = self.backend._is_peft
            self.input_device = self.backend.input_device
            return
        cls = self._model_class()
        logger.info("[VLM] loading %s → %s", self.model_name, self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        device_map: Any = "auto" if self.device in ("auto", "multi") else self.device
        if device_map == "auto":
            n = torch.cuda.device_count()
            logger.info("[VLM] sharding across %d GPU(s) with device_map=auto", n)
        from spark_vlm.experiments.internvl_backend import _dtype_kwarg

        model = cls.from_pretrained(
            self.model_name,
            device_map=device_map,
            trust_remote_code=True,
            **_dtype_kwarg(torch.bfloat16),
        )
        if self.adapter_path and Path(self.adapter_path).exists():
            from peft import PeftModel

            logger.info("[VLM] resuming adapter %s", self.adapter_path)
            model = PeftModel.from_pretrained(model, self.adapter_path, is_trainable=self.trainable)
            self._is_peft = True
        elif self.trainable:
            from peft import LoraConfig, TaskType, get_peft_model

            lc = LoraConfig(
                r=self.lora_cfg.get("r", 16),
                lora_alpha=self.lora_cfg.get("lora_alpha", 32),
                target_modules=self.lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=self.lora_cfg.get("lora_dropout", 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lc)
            self._is_peft = True
            trainable, total = (0, 0)
            for p in model.parameters():
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()
            logger.info(
                "[VLM] LoRA: %d trainable / %d total (%.4f%%)",
                trainable,
                total,
                100 * trainable / max(total, 1),
            )
        self.model = model
        self.model.eval()
        try:
            self.input_device = next(self.model.parameters()).device
        except StopIteration:
            self.input_device = self.device
        logger.info("[VLM] input device: %s", self.input_device)

    def _model_class(self):
        name = self.model_name.lower()
        try:
            if "qwen3-vl" in name:
                from transformers import Qwen3VLForConditionalGeneration as C

                return C
            if "qwen2.5-vl" in name or "qwen2_5_vl" in name:
                from transformers import Qwen2_5_VLForConditionalGeneration as C

                return C
            if "internvl" in name:
                from transformers import InternVLForConditionalGeneration as C

                return C
        except ImportError:
            pass
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq

    def _messages(self, req: GenRequest) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for ip in list(req.image_paths)[: self.max_images_per_prompt]:
            p = Path(ip)
            if p.exists() and _valid_image(p):
                content.append({"type": "image", "image": str(p.resolve())})
        content.append({"type": "text", "text": req.text})
        return [{"role": "user", "content": content}]

    def generate(
        self,
        requests: Sequence[GenRequest],
        n_samples: int = 1,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = True,
        temperature: Optional[float] = None,
    ) -> List[List[str]]:
        import torch

        if self.backend is not None:
            bs = max(1, self.gen_batch_size // max(1, n_samples))
            img_bs = max(1, bs // 4)
            out: List[List[str]] = []
            groups = _split_by_images(requests, bs, img_bs, self.max_images_per_prompt)
            for chunk in groups:
                try:
                    out.extend(
                        self.backend.generate(
                            [r.text for r in chunk],
                            [
                                [
                                    p
                                    for p in r.image_paths
                                    if Path(p).exists() and _valid_image(Path(p))
                                ][: self.max_images_per_prompt]
                                for r in chunk
                            ],
                            n_samples=n_samples,
                            max_new_tokens=max_new_tokens or self.max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature
                            if temperature is not None
                            else self.temperature,
                            top_p=self.top_p,
                        )
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    sub = max(1, len(chunk) // 2)
                    logger.warning("[VLM] OOM on %d prompts — retrying at %d", len(chunk), sub)
                    for j in range(0, len(chunk), sub):
                        piece = chunk[j : j + sub]
                        try:
                            out.extend(
                                self.backend.generate(
                                    [r.text for r in piece],
                                    [
                                        [
                                            p
                                            for p in r.image_paths
                                            if Path(p).exists() and _valid_image(Path(p))
                                        ][: self.max_images_per_prompt]
                                        for r in piece
                                    ],
                                    n_samples,
                                    max_new_tokens or self.max_new_tokens,
                                    do_sample,
                                    temperature if temperature is not None else self.temperature,
                                    self.top_p,
                                )
                            )
                        except Exception as e:
                            logger.warning("[VLM] generation failed: %s", e)
                            out.extend([[""] * n_samples for _ in piece])
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning("[VLM] batch generation failed: %s", e)
                    out.extend([[""] * n_samples for _ in chunk])
            return out
        out: List[List[str]] = [[] for _ in requests]
        idx_text = [i for i, r in enumerate(requests) if not self._usable_images(r)]
        idx_img = [i for i, r in enumerate(requests) if self._usable_images(r)]
        for group, has_img in ((idx_text, False), (idx_img, True)):
            bs = self.gen_batch_size if not has_img else max(1, self.gen_batch_size // 8)
            for start in range(0, len(group), bs):
                chunk = group[start : start + bs]
                texts, images = ([], [])
                for i in chunk:
                    msgs = self._messages(requests[i])
                    texts.append(
                        self.processor.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True
                        )
                    )
                    if has_img:
                        images.append(self._load_images(requests[i]))
                try:
                    samples = self._generate_chunk(
                        texts,
                        images if has_img else None,
                        n_samples,
                        max_new_tokens,
                        do_sample,
                        temperature,
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning("[VLM] OOM at batch %d — retrying one at a time", len(chunk))
                    torch.cuda.empty_cache()
                    samples = []
                    for k, i in enumerate(chunk):
                        try:
                            s = self._generate_chunk(
                                [texts[k]],
                                [images[k]] if has_img else None,
                                n_samples,
                                max_new_tokens,
                                do_sample,
                                temperature,
                            )
                            samples.extend(s)
                        except Exception as e:
                            logger.warning("[VLM] generation failed for one item: %s", e)
                            samples.append([""] * n_samples)
                for k, i in enumerate(chunk):
                    out[i] = samples[k] if k < len(samples) else [""] * n_samples
        return out

    def _usable_images(self, req: GenRequest) -> bool:
        return any(
            (
                Path(p).exists() and _valid_image(Path(p))
                for p in list(req.image_paths)[: self.max_images_per_prompt]
            )
        )

    def _load_images(self, req: GenRequest) -> List[Any]:
        from PIL import Image

        imgs = []
        for ip in list(req.image_paths)[: self.max_images_per_prompt]:
            p = Path(ip)
            if p.exists() and _valid_image(p):
                im = Image.open(p).convert("RGB")
                im.thumbnail((768, 768))
                imgs.append(im)
        return imgs

    def _trim_text(self, text: str) -> str:
        budget = self.max_prompt_tokens * 4
        if len(text) <= budget:
            return text
        head = int(budget * 0.6)
        tail = budget - head
        return f"{text[:head]}\n...[trimmed]...\n{text[-tail:]}"

    def _generate_chunk(
        self, texts, images, n_samples, max_new_tokens, do_sample, temperature
    ) -> List[List[str]]:
        import torch

        kwargs: Dict[str, Any] = {
            "text": texts,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.max_prompt_tokens,
        }
        if images is not None:
            kwargs["text"] = [self._trim_text(t) for t in texts]
            kwargs["images"] = images
            kwargs["truncation"] = False
            kwargs.pop("max_length")
        inputs = self.processor(**kwargs).to(self.input_device)
        plen = inputs["input_ids"].shape[1]
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=do_sample,
                temperature=(temperature if temperature is not None else self.temperature)
                if do_sample
                else None,
                top_p=self.top_p if do_sample else None,
                num_return_sequences=n_samples,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        decoded = self.processor.tokenizer.batch_decode(out[:, plen:], skip_special_tokens=True)
        return [
            [decoded[i * n_samples + j].strip() for j in range(n_samples)]
            for i in range(len(texts))
        ]

    def text_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer
        return self.processor.tokenizer

    def lm_forward(self, input_ids, attention_mask, labels):
        if self.backend is not None:
            return self.backend.lm_forward(input_ids, attention_mask, labels)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def disable_adapter(self):
        if self.backend is not None:
            return self.backend.disable_adapter()
        return self.model.disable_adapter()

    def save_adapter(self, path: str) -> Optional[str]:
        if not self._is_peft:
            return None
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        logger.info("[VLM] adapter saved → %s", path)
        return path

    def free(self) -> None:
        import torch

        if self.backend is not None:
            self.backend.model = None
            self.backend.tokenizer = None
            self.backend = None
        self.model = None
        self.processor = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def assert_fully_on_gpu(model, model_name: str = "") -> None:
        import torch

        bad = {}
        for n, prm in model.named_parameters():
            d = prm.device.type
            if d != "cuda":
                bad[d] = bad.get(d, 0) + 1
        if bad:
            raise RuntimeError(
                f"{model_name or 'model'} is not fully on GPU: {bad} parameters off-device (meta/cpu = accelerate offload). Free the previous model before loading this one."
            )
