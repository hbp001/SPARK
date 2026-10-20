from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from .question_generator import GeneratedQuestion

logger = logging.getLogger(__name__)


def _is_valid_image(path: Path) -> bool:
    try:
        if path.stat().st_size < 100:
            return False
        from PIL import Image as PILImage

        with PILImage.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


@dataclass
class GeneratedAnswer:
    question_id: str
    answer_text: str
    sample_idx: int
    used_images: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerBundle:
    question: GeneratedQuestion
    answers: List[GeneratedAnswer]
    best_answer: Optional[GeneratedAnswer] = None


class AnswerGenerator:
    def __init__(self, cfg: dict, vllm_cfg: dict, shared_vlm=None):
        self.cfg = cfg
        self.vllm_cfg = vllm_cfg
        self.n_samples = cfg.get("n_samples_per_question", 3)
        self._shared_vlm = shared_vlm
        self._model = None
        self._processor = None

    def _load_model(self) -> bool:
        if self._shared_vlm is not None:
            if not self._shared_vlm.is_loaded:
                return self._shared_vlm.load()
            return True
        if self._model is not None:
            return True
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            model_name = self.vllm_cfg.get("name", "Qwen/Qwen3-VL-4B-Instruct")
            adapter_path = self.vllm_cfg.get("_adapter_path")
            logger.info(f"[AGen] Loading VLM (standalone): {model_name}")
            base = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
            if adapter_path and Path(adapter_path).exists():
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(base, adapter_path)
            else:
                self._model = base
            self._processor = AutoProcessor.from_pretrained(model_name)
            return True
        except Exception as e:
            logger.warning(f"[AGen] VLM load failed: {e}")
            return False

    def _build_messages(self, question: GeneratedQuestion) -> List[Dict]:
        content = []
        for img_path in question.image_paths[:2]:
            p = Path(img_path)
            if p.exists() and _is_valid_image(p):
                content.append({"type": "image", "image": str(p.resolve())})
        prompt = f"[Question]\n{question.question_text}\n\nPlease reason step by step and provide your final answer."
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _infer_once(self, messages: List[Dict]) -> str:
        if self._shared_vlm is not None and self._shared_vlm.is_loaded:
            return self._shared_vlm.infer(
                messages, max_new_tokens=self.vllm_cfg.get("max_new_tokens", 512)
            )
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            first_device = next(iter(self._model.parameters())).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.vllm_cfg.get("max_new_tokens", 512),
                temperature=self.vllm_cfg.get("temperature", 0.8),
                do_sample=True,
            )
            trimmed = out[0][inputs["input_ids"].shape[1] :]
            return self._processor.decode(trimmed, skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"[AGen] Inference error: {e}")
            return ""

    def generate(self, questions: List[GeneratedQuestion]) -> List[AnswerBundle]:
        if not self._load_model():
            logger.warning("[AGen] VLM not available — using fallback answers")
            return self._fallback_bundles(questions)
        bundles: List[AnswerBundle] = []
        for q in questions:
            messages = self._build_messages(q)
            samples: List[GeneratedAnswer] = []
            for i in range(self.n_samples):
                answer_text = self._infer_once(messages)
                if not answer_text:
                    continue
                samples.append(
                    GeneratedAnswer(
                        question_id=q.question_id,
                        answer_text=answer_text,
                        sample_idx=i,
                        used_images=q.image_paths,
                    )
                )
            if samples:
                bundles.append(AnswerBundle(question=q, answers=samples))
        logger.info(f"[AGen] {len(bundles)} answer bundles generated")
        return bundles

    def _fallback_bundles(self, questions: List[GeneratedQuestion]) -> List[AnswerBundle]:
        bundles = []
        for q in questions:
            ans = GeneratedAnswer(
                question_id=q.question_id,
                answer_text=f"[Fallback] {q.question_text[:80]}",
                sample_idx=0,
                used_images=q.image_paths,
                metadata={"fallback": True},
            )
            bundles.append(AnswerBundle(question=q, answers=[ans]))
        return bundles
