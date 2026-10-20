from __future__ import annotations
import logging
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
from spark_vlm.knowledge_graph.graph_store import NetworkXGraphStore
from spark_vlm.knowledge_graph.schema import KGNode
from .answer_generator import AnswerBundle, GeneratedAnswer

logger = logging.getLogger(__name__)


@dataclass
class RewardScore:
    question_id: str
    answer_idx: int
    r_correctness: float
    r_path: float
    r_consistency: float
    r_challenger: float
    total: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class KGVerifier:
    def __init__(self, store: NetworkXGraphStore):
        self.store = store

    def _clean_gold_answer(self, text: str) -> str:
        import re

        text = re.sub("<[^>]+>", " ", text)
        text = re.sub("\\s+", " ", text).strip()
        return text

    def extract_gold_answer(self, bundle: AnswerBundle) -> Optional[str]:
        path = bundle.question.reasoning_path
        nodes = path.nodes
        if not nodes:
            return None
        target_node = nodes[-1]
        edge_types = path.edge_types
        last_edge = edge_types[-1] if edge_types else ""
        if any((k in last_edge for k in ["QUANTIFIES", "MEASURES"])):
            nums = re.findall("\\b\\d+\\.?\\d*\\b", target_node.content or "")
            return nums[0] if nums else (target_node.content or "")[:100]
        if "DEFINES" in last_edge:
            sentences = (target_node.content or "").split(".")
            return sentences[0].strip() if sentences else (target_node.content or "")[:100]
        if "SAME_CONCEPT" in last_edge and len(nodes) >= 2:
            return self._extract_common_concept(nodes[-2], nodes[-1])
        raw = (target_node.content or "")[:400].strip()
        return self._clean_gold_answer(raw)

    def _extract_common_concept(self, node_a: KGNode, node_b: KGNode) -> str:
        words_a = set(re.findall("\\b[a-zA-Z]{4,}\\b", (node_a.content or "").lower()))
        words_b = set(re.findall("\\b[a-zA-Z]{4,}\\b", (node_b.content or "").lower()))
        common = words_a & words_b
        return " ".join(sorted(common)[:5]) if common else (node_a.content or "")[:100]

    def verify(self, answer_text: str, gold_answer: str) -> float:
        if not gold_answer or not answer_text:
            return 0.0
        ans = answer_text.lower().strip()
        gold = gold_answer.lower().strip()
        if gold in ans or ans in gold:
            return 1.0
        gold_nums = set(re.findall("\\b\\d+\\.?\\d*\\b", gold))
        ans_nums = set(re.findall("\\b\\d+\\.?\\d*\\b", ans))
        if gold_nums and gold_nums == ans_nums:
            return 1.0
        if gold_nums and ans_nums and gold_nums & ans_nums:
            return 0.7
        gold_words = set(re.findall("\\b[a-zA-Z]{4,}\\b", gold))
        ans_words = set(re.findall("\\b[a-zA-Z]{4,}\\b", ans))
        if gold_words:
            overlap = len(gold_words & ans_words) / len(gold_words)
            if overlap >= 0.7:
                return 0.8
            if overlap >= 0.4:
                return 0.5
        sim = SequenceMatcher(None, ans, gold).ratio()
        return sim if sim >= 0.5 else 0.0

    def get_edge_type_accuracy(
        self, bundles: List[AnswerBundle], correctness_map: Dict[str, float]
    ) -> Dict[str, float]:
        edge_scores: Dict[str, List[float]] = {}
        for bundle in bundles:
            qid = bundle.question.question_id
            edge_types = bundle.question.reasoning_path.edge_types
            if not edge_types:
                continue
            primary = edge_types[0]
            score = correctness_map.get(qid, 0.0)
            edge_scores.setdefault(primary, []).append(score)
        return {et: round(sum(s) / len(s), 4) for et, s in edge_scores.items()}


class RewardCalculator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.w_correctness = cfg.get("r_correctness_weight", 0.6)
        self.w_path = cfg.get("r_path_weight", 0.2)
        self.w_consistency = cfg.get("r_consistency_weight", 0.2)
        self.edge_type_accuracy: Dict[str, float] = {}

    def update_weights_for_epoch(self, epoch: int, total_epochs: int) -> None:
        progress = epoch / max(total_epochs - 1, 1)
        self.w_correctness = 0.5 + 0.2 * progress
        self.w_path = 0.3 - 0.1 * progress
        self.w_consistency = 0.2 - 0.1 * progress
        logger.debug(
            f"[Reward] Weights: correctness={self.w_correctness:.2f}, path={self.w_path:.2f}, consistency={self.w_consistency:.2f}"
        )

    def _challenger_reward(self, correctness_list: List[float]) -> float:
        k = len(correctness_list)
        if k == 0:
            return 0.0
        mean_c = sum(correctness_list) / k
        variance = mean_c * (1.0 - mean_c)
        return round(math.exp(-((variance - 0.25) ** 2) / 0.02), 4)

    def _r_correctness(
        self, answer: GeneratedAnswer, gold_answer: Optional[str], verifier: KGVerifier
    ) -> float:
        if gold_answer is None:
            return 0.0
        return verifier.verify(answer.answer_text, gold_answer)

    def _r_path(self, bundle: AnswerBundle, store: NetworkXGraphStore) -> float:
        path = bundle.question.reasoning_path
        if not path.node_ids or len(path.node_ids) < 2:
            return 0.0
        total = len(path.node_ids) - 1
        weight_sum = 0.0
        for i in range(total):
            src, dst = (path.node_ids[i], path.node_ids[i + 1])
            if not store._g.has_edge(src, dst):
                continue
            edge_data = store._g.get_edge_data(src, dst) or {}
            if "weight" in edge_data:
                weight = float(edge_data["weight"])
            else:
                inner = next(iter(edge_data.values()), {})
                weight = float(inner.get("weight", 1.0)) if isinstance(inner, dict) else 1.0
            weight_sum += weight
        return round(weight_sum / total, 4)

    def _r_consistency(self, answer: GeneratedAnswer, bundle: AnswerBundle) -> float:
        ans_text = answer.answer_text
        ans_nums = set(re.findall("\\b\\d+\\.?\\d*\\b", ans_text))
        kg_text = ""
        kg_nums: set = set()
        for node in bundle.question.reasoning_path.nodes:
            if node.content:
                kg_nums.update(re.findall("\\b\\d+\\.?\\d*\\b", node.content))
                kg_text += " " + node.content[:200]
        scores: List[float] = []
        weights: List[float] = []
        if ans_nums and kg_nums:
            scores.append(len(ans_nums & kg_nums) / len(ans_nums))
            weights.append(0.6)
        key_words = re.findall("\\b[a-zA-Z]{4,}\\b", kg_text.lower())[:20]
        if key_words:
            ans_lower = ans_text.lower()
            scores.append(sum((1 for w in key_words if w in ans_lower)) / len(key_words))
            weights.append(0.4)
        if not scores:
            return 0.0
        total_w = sum(weights)
        return round(min(sum((s * w for s, w in zip(scores, weights))) / total_w, 1.0), 4)

    def calculate(
        self, bundles: List[AnswerBundle], store: NetworkXGraphStore
    ) -> List[AnswerBundle]:
        verifier = KGVerifier(store)
        correctness_map: Dict[str, float] = {}
        for bundle in bundles:
            if not bundle.answers:
                continue
            gold_answer = verifier.extract_gold_answer(bundle)
            if gold_answer is None:
                logger.debug(f"[Reward] {bundle.question.question_id}: skipped — no gold answer")
                continue
            r_path_score = self._r_path(bundle, store)
            correctness_list: List[float] = []
            scored: List[Tuple[float, GeneratedAnswer]] = []
            for i, answer in enumerate(bundle.answers):
                r_corr = self._r_correctness(answer, gold_answer, verifier)
                r_cons = self._r_consistency(answer, bundle)
                total = (
                    self.w_correctness * r_corr
                    + self.w_path * r_path_score
                    + self.w_consistency * r_cons
                )
                answer.metadata["reward"] = RewardScore(
                    question_id=bundle.question.question_id,
                    answer_idx=i,
                    r_correctness=round(r_corr, 4),
                    r_path=round(r_path_score, 4),
                    r_consistency=round(r_cons, 4),
                    r_challenger=0.0,
                    total=round(total, 4),
                    metadata={
                        "gold_answer": gold_answer,
                        "edge_types": bundle.question.reasoning_path.edge_types,
                    },
                )
                correctness_list.append(r_corr)
                scored.append((total, answer))
            r_challenger = self._challenger_reward(correctness_list)
            for _, answer in scored:
                answer.metadata["reward"].r_challenger = r_challenger
            scored.sort(key=lambda x: x[0], reverse=True)
            bundle.best_answer = scored[0][1]
            qid = bundle.question.question_id
            correctness_map[qid] = sum(correctness_list) / len(correctness_list)
            logger.debug(
                f"[Reward] {qid}: best={scored[0][0]:.3f} r_C={r_challenger:.3f} gold='{(gold_answer or '')[:50]}'"
            )
        self.edge_type_accuracy = verifier.get_edge_type_accuracy(bundles, correctness_map)
        if self.edge_type_accuracy:
            logger.info(
                "[Reward] Edge-type accuracy: "
                + ", ".join(
                    (
                        f"{et}={acc:.2f}"
                        for et, acc in sorted(self.edge_type_accuracy.items(), key=lambda x: x[1])
                    )
                )
            )
        return bundles

    def filter_high_reward(
        self, bundles: List[AnswerBundle], threshold: float = 0.5
    ) -> List[AnswerBundle]:
        filtered = [
            b
            for b in bundles
            if b.best_answer
            and b.best_answer.metadata.get("reward")
            and (b.best_answer.metadata["reward"].total >= threshold)
        ]
        logger.info(
            f"[Reward] Filtered: {len(filtered)}/{len(bundles)} above threshold={threshold}"
        )
        return filtered
