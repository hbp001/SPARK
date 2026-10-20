from __future__ import annotations
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from spark_vlm.knowledge_graph.graph_store import NetworkXGraphStore
from spark_vlm.knowledge_graph.schema import EdgeType, KGEdge, KGNode, NodeType
from .answer_generator import AnswerBundle

logger = logging.getLogger(__name__)
CONFIDENCE_BOOST = 0.05
CONFIDENCE_DECAY = 0.1
MIN_CONFIDENCE = 0.1
PRUNE_THRESHOLD = 0.15
MERGE_THRESHOLD = 0.88


class KGRefiner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.prune_threshold = cfg.get("prune_threshold", PRUNE_THRESHOLD)
        self.merge_threshold = cfg.get("merge_threshold", MERGE_THRESHOLD)

    def refine(
        self, store: NetworkXGraphStore, bundles: List[AnswerBundle], reward_threshold: float = 0.5
    ) -> Dict[str, int]:
        stats = {"edges_added": 0, "edges_pruned": 0, "confidence_updated": 0, "nodes_merged": 0}
        for bundle in bundles:
            if not bundle.best_answer:
                continue
            reward = bundle.best_answer.metadata.get("reward")
            if not reward:
                continue
            is_correct = reward.total >= reward_threshold
            path = bundle.question.reasoning_path
            updated = self._update_edge_confidence(store, path, is_correct)
            stats["confidence_updated"] += updated
            if is_correct and reward.r_path < 0.5:
                added = self._detect_and_add_missing_edges(store, bundle, reward.total)
                stats["edges_added"] += added
        pruned = self._prune_spurious_edges(store)
        stats["edges_pruned"] += pruned
        logger.info(
            f"[KGRefiner] Refinement complete: +{stats['edges_added']} edges, -{stats['edges_pruned']} pruned, {stats['confidence_updated']} updated, {stats['nodes_merged']} merged"
        )
        return stats

    def _update_edge_confidence(self, store: NetworkXGraphStore, path, is_correct: bool) -> int:
        updated = 0
        node_ids = path.node_ids
        for i in range(len(node_ids) - 1):
            src, dst = (node_ids[i], node_ids[i + 1])
            if not store._g.has_edge(src, dst):
                continue
            for key, data in store._g[src][dst].items():
                current = float(data.get("confidence", 0.5))
                if is_correct:
                    new_conf = min(1.0, current + CONFIDENCE_BOOST)
                else:
                    new_conf = max(MIN_CONFIDENCE, current - CONFIDENCE_DECAY)
                store._g[src][dst][key]["confidence"] = round(new_conf, 4)
                updated += 1
        return updated

    def _detect_and_add_missing_edges(
        self, store: NetworkXGraphStore, bundle: AnswerBundle, reward: float
    ) -> int:
        added = 0
        path = bundle.question.reasoning_path
        nodes = path.nodes
        if len(nodes) < 2:
            return 0
        for i in range(len(nodes) - 1):
            src = nodes[i]
            dst = nodes[i + 1]
            if store._g.has_edge(src.node_id, dst.node_id):
                continue
            inferred_type = self._infer_edge_type(src, dst)
            new_edge = KGEdge(
                src_id=src.node_id,
                dst_id=dst.node_id,
                edge_type=inferred_type,
                weight=round(reward * 0.8, 4),
                confidence=round(reward * 0.7, 4),
                metadata={"method": "self_play_inference", "reward": reward, "inferred": True},
            )
            store.add_edge(new_edge)
            added += 1
            logger.debug(
                f"[KGRefiner] Inferred edge: {src.node_id} --[{inferred_type.value}]--> {dst.node_id}"
            )
        return added

    def _infer_edge_type(self, src: KGNode, dst: KGNode) -> EdgeType:
        s, d = (src.node_type, dst.node_type)
        if d in (NodeType.FIGURE, NodeType.TABLE):
            return EdgeType.ILLUSTRATES
        if d == NodeType.EQUATION:
            return EdgeType.QUANTIFIES
        if d == NodeType.CONCEPT:
            return EdgeType.DEFINES
        if s == NodeType.CLAIM:
            return EdgeType.SUPPORTS
        return EdgeType.SUPPORTS

    def _prune_spurious_edges(self, store: NetworkXGraphStore) -> int:
        to_remove: List[Tuple] = []
        for src, dst, key, data in store._g.edges(data=True, keys=True):
            method = data.get("metadata", {}).get("method", "")
            if "self_play_inference" not in str(method):
                continue
            confidence = float(data.get("confidence", 1.0))
            if confidence <= self.prune_threshold:
                to_remove.append((src, dst, key))
        for src, dst, key in to_remove:
            store._g.remove_edge(src, dst, key=key)
            logger.debug(f"[KGRefiner] Pruned edge: {src} → {dst}")
        return len(to_remove)
