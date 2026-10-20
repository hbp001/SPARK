from __future__ import annotations
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from spark_vlm.knowledge_graph.schema import EdgeType, KGNode, NodeType
from spark_vlm.knowledge_graph.graph_store import NetworkXGraphStore

logger = logging.getLogger(__name__)
EDGE_TO_DIFFICULTY: Dict[str, int] = {
    "CONTAINS": 1,
    "HAS_CAPTION": 1,
    "DEFINES": 2,
    "QUANTIFIES": 2,
    "SUPPORTS": 2,
    "ILLUSTRATES": 2,
    "REFERENCES": 2,
    "DERIVES_FROM": 3,
    "COMPARES": 3,
    "CONTRADICTS": 3,
    "SAME_CONCEPT": 4,
    "CROSS_DOC": 4,
}
EDGE_TO_QTYPE: Dict[str, str] = {
    "CONTAINS": "VQA",
    "HAS_CAPTION": "VQA",
    "DEFINES": "VQA",
    "QUANTIFIES": "VQA",
    "SUPPORTS": "Reasoning",
    "ILLUSTRATES": "VQA",
    "REFERENCES": "Reasoning",
    "DERIVES_FROM": "Reasoning",
    "COMPARES": "MCQ",
    "CONTRADICTS": "Reasoning",
    "SAME_CONCEPT": "InstructionFollowing",
    "CROSS_DOC": "InstructionFollowing",
}
_EDGE_PROMPTS: Dict[str, str] = {
    "SUPPORTS": "You are a scientific question generator with access to a Knowledge Graph.\n\nKG Path: [{src_type}] → SUPPORTS → [{dst_type}]\nSource content: {src_content}\nTarget content (the claim being supported): {dst_content}\n{image_hint}\n\nGenerate ONE question that requires understanding WHY the source supports the target claim.\nThe question must be answerable from the content without seeing this KG path.\nGold answer (for verification): {gold_answer}\n\nOutput format: Question only, no explanation.\nQuestion:",
    "CONTRADICTS": "You are a scientific question generator with access to a Knowledge Graph.\n\nKG Path: [{src_type}] → CONTRADICTS → [{dst_type}]\nContent A: {src_content}\nContent B (contradicting): {dst_content}\n{image_hint}\n\nGenerate ONE question that requires identifying the contradiction or difference between A and B.\nGold answer (for verification): {gold_answer}\n\nOutput format: Question only.\nQuestion:",
    "COMPARES": "You are a scientific question generator with access to a Knowledge Graph.\n\nKG Path: [{src_type}] → COMPARES → [{dst_type}]\nMethod/Result A: {src_content}\nMethod/Result B: {dst_content}\n{image_hint}\n\nGenerate ONE multiple-choice question comparing A and B with 4 options (A/B/C/D).\nGold answer (for verification): {gold_answer}\n\nOutput format:\nQuestion: <text>\nA) <option>\nB) <option>\nC) <option>\nD) <option>\nAnswer: <letter>",
    "QUANTIFIES": "You are a scientific question generator with access to a Knowledge Graph.\n\nKG Path: [{src_type}] → QUANTIFIES → [{dst_type}]\nMeasurement context: {src_content}\nQuantified result: {dst_content}\n{image_hint}\n\nGenerate ONE question about the numerical result or measurement shown.\nGold answer (for verification): {gold_answer}\n\nOutput format: Question only.\nQuestion:",
    "SAME_CONCEPT": "You are a scientific question generator with access to a Knowledge Graph.\n\nCross-document KG connection:\nPaper A concept: {src_content}\nPaper B concept (same concept, different expression): {dst_content}\n\nGenerate ONE question requiring synthesis across both papers about this concept.\nGold answer (for verification): {gold_answer}\n\nOutput format: Question only.\nQuestion:",
    "DEFAULT": "You are a scientific question generator with access to a Knowledge Graph.\n\nKG Path: [{src_type}] --[{edge_type}]--> [{dst_type}]\nContent: {src_content}\nRelated content: {dst_content}\n{image_hint}\n\nGenerate ONE question based on this KG relationship.\nDifficulty level: {difficulty}\nGold answer (for verification): {gold_answer}\n\nOutput format: Question only.\nQuestion:",
}
_SINGLE_NODE_PROMPT = "You are a scientific question generator with access to a Knowledge Graph.\n\nNode type: [{node_type}]\nContent: {content}\n{image_hint}\n\nGenerate ONE basic factual question about this content.\nThe answer should be directly findable in the content.\nGold answer (for verification): {gold_answer}\n\nOutput format: Question only.\nQuestion:"


@dataclass
class ReasoningPath:
    node_ids: List[str]
    edge_types: List[str]
    nodes: List[KGNode]
    difficulty: int
    gold_answer: str = ""


@dataclass
class GeneratedQuestion:
    question_id: str
    question_text: str
    question_type: str
    difficulty: int
    context_text: str
    image_paths: List[str]
    reasoning_path: ReasoningPath
    gold_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuestionGenerator:
    def __init__(self, cfg: dict, vllm_cfg: dict, shared_vlm=None):
        self.cfg = cfg
        self.vllm_cfg = vllm_cfg
        self.max_hops = cfg.get("max_hops", 3)
        self.difficulty_levels = cfg.get("difficulty_levels", 4)
        self._q_counter = 0
        self._shared_vlm = shared_vlm
        self._model = None
        self._processor = None
        self.edge_type_accuracy: Dict[str, float] = {}

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
            logger.info(f"[QGen] Loading VLM (standalone): {model_name}")
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
            logger.warning(f"[QGen] VLM load failed: {e}")
            return False

    def reload_adapter(self, adapter_path: str) -> None:
        if self._shared_vlm is not None:
            self._shared_vlm.update_adapter(adapter_path)
        else:
            self._model = None
            self._processor = None
            self.vllm_cfg = {**self.vllm_cfg, "_adapter_path": adapter_path}
        logger.info(f"[QGen] Adapter updated: {adapter_path}")

    def _infer(self, messages: List[Dict]) -> str:
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
            logger.warning(f"[QGen] Inference error: {e}")
            return ""

    def _sample_difficulty_by_curriculum(self, epoch: int, total_epochs: int) -> int:
        progress = epoch / max(total_epochs - 1, 1)
        w1 = max(0.1, 0.7 - 0.6 * progress)
        w4 = min(0.4, 0.0 + 0.4 * progress)
        remaining = 1.0 - w1 - w4
        w2 = remaining * 0.55
        w3 = remaining * 0.45
        return random.choices([1, 2, 3, 4], weights=[w1, w2, w3, w4], k=1)[0]

    def get_curriculum_stats(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        progress = epoch / max(total_epochs - 1, 1)
        w1 = max(0.1, 0.7 - 0.6 * progress)
        w4 = min(0.4, 0.0 + 0.4 * progress)
        remaining = 1.0 - w1 - w4
        w2 = remaining * 0.55
        w3 = remaining * 0.45
        return {"L1": round(w1, 2), "L2": round(w2, 2), "L3": round(w3, 2), "L4": round(w4, 2)}

    def update_edge_accuracy(self, edge_type_accuracy: Dict[str, float]) -> None:
        self.edge_type_accuracy = edge_type_accuracy
        if edge_type_accuracy:
            weak = [et for et, acc in edge_type_accuracy.items() if acc < 0.4]
            logger.info(f"[QGen] Weak edge types (acc<0.4): {weak}")

    def _get_high_value_edge_types(self) -> List[str]:
        if not self.edge_type_accuracy:
            return list(EDGE_TO_DIFFICULTY.keys())
        return sorted(
            self.edge_type_accuracy.keys(), key=lambda et: self.edge_type_accuracy.get(et, 0.5)
        )

    def sample_reasoning_paths(
        self,
        store: NetworkXGraphStore,
        paper_id: Optional[str] = None,
        n_paths: int = 20,
        epoch: int = 0,
        total_epochs: int = 1,
    ) -> List[ReasoningPath]:
        curriculum_dist = self.get_curriculum_stats(epoch, total_epochs)
        logger.info(
            f"[QGen] Curriculum (epoch {epoch + 1}/{total_epochs}): L1={curriculum_dist['L1']:.0%} L2={curriculum_dist['L2']:.0%} L3={curriculum_dist['L3']:.0%} L4={curriculum_dist['L4']:.0%}"
        )
        if self.edge_type_accuracy:
            weak = {et: acc for et, acc in self.edge_type_accuracy.items() if acc < 0.4}
            if weak:
                logger.info(f"[QGen] Boosting weak edge types: {weak}")
        priority_types = [
            NodeType.FIGURE,
            NodeType.TABLE,
            NodeType.CONCEPT,
            NodeType.CLAIM,
            NodeType.TEXT_BLOCK,
        ]
        candidate_nodes: List[KGNode] = []
        for ntype in priority_types:
            nodes = store.get_nodes_by_type(ntype)
            if paper_id:
                nodes = [n for n in nodes if n.paper_id == paper_id]
            candidate_nodes.extend(nodes)
        if not candidate_nodes:
            logger.warning(f"[QGen] No candidate nodes for paper={paper_id}")
            return []
        paths: List[ReasoningPath] = []
        for _ in range(n_paths):
            start = random.choice(candidate_nodes)
            difficulty = self._sample_difficulty_by_curriculum(epoch, total_epochs)
            if difficulty == 1:
                gold = (start.content or "")[:150].strip()
                paths.append(
                    ReasoningPath(
                        node_ids=[start.node_id],
                        edge_types=[],
                        nodes=[start],
                        difficulty=1,
                        gold_answer=gold,
                    )
                )
                continue
            max_hops = min(difficulty, self.max_hops)
            weak_types = (
                [et for et, acc in self.edge_type_accuracy.items() if acc < 0.4]
                if self.edge_type_accuracy
                else []
            )
            walked = self._walk_with_edge_preference(
                store, start.node_id, max_hops, weak_types, difficulty
            )
            walked_nodes = [store.get_node(nid) for nid in walked if store.get_node(nid)]
            if len(walked_nodes) < 2:
                continue
            edge_types: List[str] = []
            for i in range(len(walked) - 1):
                edges = store._g.get_edge_data(walked[i], walked[i + 1])
                if edges:
                    et = list(edges.values())[0].get("edge_type", "CONTAINS")
                    edge_types.append(et)
            actual_difficulty = (
                max((EDGE_TO_DIFFICULTY.get(et, difficulty) for et in edge_types))
                if edge_types
                else difficulty
            )
            gold = self._extract_gold_answer(walked_nodes, edge_types)
            paths.append(
                ReasoningPath(
                    node_ids=walked,
                    edge_types=edge_types,
                    nodes=walked_nodes,
                    difficulty=actual_difficulty,
                    gold_answer=gold,
                )
            )
        diff_dist = {1: 0, 2: 0, 3: 0, 4: 0}
        for p in paths:
            diff_dist[p.difficulty] = diff_dist.get(p.difficulty, 0) + 1
        logger.info(f"[QGen] Sampled {len(paths)} paths — actual dist: {diff_dist}")
        return paths

    def _walk_with_edge_preference(
        self,
        store: NetworkXGraphStore,
        start_id: str,
        max_hops: int,
        preferred_edge_types: List[str],
        difficulty: int,
    ) -> List[str]:
        path = [start_id]
        for _ in range(max_hops):
            current = path[-1]
            neighbors = list(store._g.successors(current))
            if not neighbors:
                break
            preferred = []
            for nb in neighbors:
                edges = store._g.get_edge_data(current, nb)
                if edges:
                    et = list(edges.values())[0].get("edge_type", "")
                    if et in preferred_edge_types:
                        preferred.append(nb)
            if difficulty == 4:
                cross_doc = []
                for nb in neighbors:
                    src_node = store.get_node(current)
                    dst_node = store.get_node(nb)
                    if src_node and dst_node and (src_node.paper_id != dst_node.paper_id):
                        cross_doc.append(nb)
                if cross_doc:
                    path.append(random.choice(cross_doc))
                    continue
            chosen = preferred if preferred else neighbors
            path.append(random.choice(chosen))
        return path

    def _extract_gold_answer(self, nodes: List[KGNode], edge_types: List[str]) -> str:
        if not nodes:
            return ""
        target = nodes[-1]
        last_edge = edge_types[-1] if edge_types else ""
        import re

        if any((k in last_edge for k in ["QUANTIFIES", "MEASURES"])):
            nums = re.findall("\\b\\d+\\.?\\d*\\b", target.content or "")
            return nums[0] if nums else (target.content or "")[:100]
        if "DEFINES" in last_edge:
            sentences = (target.content or "").split(".")
            return sentences[0].strip() if sentences else (target.content or "")[:100]
        if "SAME_CONCEPT" in last_edge and len(nodes) >= 2:
            words_a = set(re.findall("\\b[a-zA-Z]{4,}\\b", (nodes[-2].content or "").lower()))
            words_b = set(re.findall("\\b[a-zA-Z]{4,}\\b", (target.content or "").lower()))
            common = words_a & words_b
            return " ".join(sorted(common)[:5]) if common else (target.content or "")[:100]
        return (target.content or "").strip()

    def generate_questions(self, paths: List[ReasoningPath]) -> List[GeneratedQuestion]:
        if not self._load_model():
            logger.warning("[QGen] VLM not available — using template questions")
            return self._fallback_questions(paths)
        questions: List[GeneratedQuestion] = []
        for path in paths:
            q = self._generate_one(path)
            if q:
                questions.append(q)
        logger.info(f"[QGen] Generated {len(questions)} questions")
        return questions

    def _generate_one(self, path: ReasoningPath) -> Optional[GeneratedQuestion]:
        image_paths = self._collect_images(path)
        image_hint = f"[Image available: {image_paths[0]}]" if image_paths else ""
        has_figure_node = any((n.node_type == NodeType.FIGURE for n in path.nodes))
        if has_figure_node and (not image_paths):
            logger.debug("[QGen] Skipping figure question — no image available")
            return None
        if not path.edge_types:
            node = path.nodes[0]
            prompt = _SINGLE_NODE_PROMPT.format(
                node_type=node.node_type.value,
                content=(node.content or "")[:600],
                image_hint=image_hint,
                gold_answer=path.gold_answer,
            )
        else:
            primary_edge = path.edge_types[0]
            src = path.nodes[0]
            dst = path.nodes[-1]
            template = _EDGE_PROMPTS.get(primary_edge, _EDGE_PROMPTS["DEFAULT"])
            prompt = template.format(
                src_type=src.node_type.value,
                dst_type=dst.node_type.value,
                src_content=(src.content or "")[:400],
                dst_content=(dst.content or "")[:400],
                edge_type=primary_edge,
                difficulty=path.difficulty,
                image_hint=image_hint,
                gold_answer=path.gold_answer,
            )
        if image_paths and Path(image_paths[0]).exists():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(Path(image_paths[0]).resolve())},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        q_text = self._infer(messages)
        if not q_text:
            return None
        self._q_counter += 1
        edge_types = path.edge_types
        primary = edge_types[0] if edge_types else "CONTAINS"
        q_type = EDGE_TO_QTYPE.get(primary, "VQA")
        return GeneratedQuestion(
            question_id=f"q_{self._q_counter:06d}",
            question_text=q_text,
            question_type=q_type,
            difficulty=path.difficulty,
            context_text=self._build_context(path),
            image_paths=image_paths,
            reasoning_path=path,
            gold_answer=path.gold_answer,
            metadata={
                "paper_ids": list({n.paper_id for n in path.nodes}),
                "edge_types": edge_types,
                "primary_edge": primary,
            },
        )

    def _build_context(self, path: ReasoningPath) -> str:
        parts = []
        for node in path.nodes:
            if node.content and node.content.strip():
                parts.append(f"[{node.node_type.value}] {node.content[:300]}")
        return "\n\n".join(parts)

    def _collect_images(self, path: ReasoningPath) -> List[str]:
        imgs = []
        for node in path.nodes:
            if node.image_path and Path(node.image_path).exists():
                imgs.append(node.image_path)
        return imgs

    def _map_type(self, difficulty: int) -> str:
        return {1: "VQA", 2: "MCQ", 3: "Reasoning", 4: "InstructionFollowing"}.get(
            difficulty, "VQA"
        )

    def _fallback_questions(self, paths: List[ReasoningPath]) -> List[GeneratedQuestion]:
        templates = {
            "SUPPORTS": "What evidence does this content provide to support the main claim?",
            "CONTRADICTS": "What is the key contradiction between these two findings?",
            "COMPARES": "Which of the following best describes the comparison shown?\nA) Option based on src\nB) Option based on dst\nC) Both are equal\nD) Cannot be determined",
            "QUANTIFIES": "What is the numerical value or measurement shown?",
            "DEFINES": "How is this concept defined in the context?",
            "SAME_CONCEPT": "How do the two papers differently describe this concept?",
            "DEFAULT_L1": "What does this {type} show or describe?",
            "DEFAULT_L2": "What is the relationship shown between these elements?",
            "DEFAULT_L3": "How does the information connect across these content pieces?",
            "DEFAULT_L4": "Based on this content, propose a follow-up experiment or synthesis.",
        }
        questions = []
        for path in paths:
            self._q_counter += 1
            primary = path.edge_types[0] if path.edge_types else ""
            node = path.nodes[0] if path.nodes else None
            ntype = node.node_type.value if node else "content"
            if primary in templates:
                q_text = templates[primary]
            else:
                key = f"DEFAULT_L{path.difficulty}"
                q_text = templates.get(key, "What can be inferred from this content?")
                q_text = q_text.format(type=ntype)
            q_type = EDGE_TO_QTYPE.get(primary, self._map_type(path.difficulty))
            questions.append(
                GeneratedQuestion(
                    question_id=f"q_{self._q_counter:06d}",
                    question_text=q_text,
                    question_type=q_type,
                    difficulty=path.difficulty,
                    context_text=self._build_context(path),
                    image_paths=self._collect_images(path),
                    reasoning_path=path,
                    gold_answer=path.gold_answer,
                    metadata={"fallback": True, "edge_types": path.edge_types},
                )
            )
        return questions
