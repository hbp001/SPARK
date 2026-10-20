from __future__ import annotations
import json
import logging
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from spark_vlm.experiments.batched_vlm import BatchedVLM, GenRequest

logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    model_name: str = "OpenGVLab/InternVL3_5-38B-Pretrained"
    device: str = "auto"
    n_epochs: int = 8
    questions_per_epoch: int = 96
    n_samples_per_question: int = 6
    reward_threshold: float = 0.35
    learning_rate: float = 5e-05
    warmup_frac: float = 0.1
    collapse_ratio: float = 0.35
    kl_coeff: float = 0.05
    grad_accum: int = 4
    max_grad_norm: float = 1.0
    train_micro_batch: int = 2
    max_groups_per_epoch: int = 0
    gen_batch_size: int = 32
    max_new_tokens: int = 320
    temperature: float = 0.9
    question_max_new_tokens: int = 96
    lora: Dict[str, Any] = field(
        default_factory=lambda: {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
        }
    )
    max_hops: int = 3
    difficulty_levels: int = 4
    kg_refine: bool = True
    edge_feedback: bool = True
    reward_module_path: str = ""
    objective: str = "spark"
    e1: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    save_every: int = 0
    log_samples: int = 4


class SelfPlayDriver:
    def __init__(
        self,
        cfg: SelfPlayConfig,
        store,
        out_dir: Path,
        paper_ids: Optional[Sequence[str]] = None,
        question_source: Optional[Callable] = None,
    ):
        self.cfg = cfg
        self.store = store
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.paper_ids = list(paper_ids) if paper_ids else None
        self.question_source = question_source
        random.seed(cfg.seed)
        sp_cfg = self._sp_cfg()
        from spark_vlm.self_play.question_generator import QuestionGenerator
        from spark_vlm.self_play.kg_refiner import KGRefiner

        self.q_gen = QuestionGenerator(sp_cfg, {"name": cfg.model_name})
        self.refiner = KGRefiner(sp_cfg)
        self.reward = self._load_reward(sp_cfg)
        self.vlm = BatchedVLM(
            model_name=cfg.model_name,
            device=cfg.device,
            lora_cfg=cfg.lora,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            gen_batch_size=cfg.gen_batch_size,
            trainable=True,
        )
        self._optimizer = None
        self._paper_node_cache: Dict[str, List[Any]] = {}
        self.history: List[Dict[str, Any]] = []
        self._allowed = set(self.paper_ids) if self.paper_ids else None
        if self._allowed:
            logger.info("[driver] path sampling restricted to %d papers", len(self._allowed))

    def _sp_cfg(self) -> Dict[str, Any]:
        c = self.cfg
        return {
            "max_hops": c.max_hops,
            "difficulty_levels": c.difficulty_levels,
            "n_samples_per_question": c.n_samples_per_question,
            "reward_threshold": c.reward_threshold,
            "lora": c.lora,
            "learning_rate": c.learning_rate,
            "reward_module_path": c.reward_module_path,
            "e1": c.e1,
            "prune_threshold": 0.15,
            "merge_threshold": 0.88,
        }

    def _load_reward(self, sp_cfg: Dict[str, Any]):
        from spark_vlm.self_play.self_play_loop import SelfPlayLoop

        return SelfPlayLoop._load_reward_calculator(sp_cfg)

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        logger.info("=" * 66)
        logger.info(
            "self-play: %d epochs x %d questions x G=%d  model=%s",
            self.cfg.n_epochs,
            self.cfg.questions_per_epoch,
            self.cfg.n_samples_per_question,
            self.cfg.model_name,
        )
        logger.info("=" * 66)
        d0 = self.adapter_delta()
        logger.info("[driver] adapter before training: %s", d0)
        best_reward, best_epoch, collapsed = (-1.0, 0, False)
        best_path = str(self.out_dir / "adapter_best")
        for epoch in range(self.cfg.n_epochs):
            te = time.time()
            self.reward.update_weights_for_epoch(epoch, self.cfg.n_epochs)
            stats = self._run_epoch(epoch)
            stats["epoch"] = epoch + 1
            stats["elapsed_s"] = round(time.time() - te, 1)
            self.history.append(stats)
            logger.info(
                "[epoch %d/%d] %.0fs | Q=%d groups=%d updated=%d skipped_flat=%d | mean_R=%.3f mean_Racc=%.3f mean_Rpath=%.3f | loss=%s",
                epoch + 1,
                self.cfg.n_epochs,
                stats["elapsed_s"],
                stats.get("n_questions", 0),
                stats.get("n_groups", 0),
                stats.get("n_updates", 0),
                stats.get("n_skipped_zero_variance", 0),
                stats.get("mean_total_reward") or 0.0,
                stats.get("mean_r_correctness") or 0.0,
                stats.get("mean_r_path") or 0.0,
                stats.get("mean_loss"),
            )
            r_now = stats.get("mean_total_reward") or 0.0
            if r_now > best_reward:
                best_reward = r_now
                best_epoch = epoch + 1
                self.vlm.save_adapter(best_path)
                logger.info(
                    "[driver] new best epoch %d (mean_R=%.3f) → snapshot", best_epoch, best_reward
                )
            elif best_reward > 0 and r_now < self.cfg.collapse_ratio * best_reward:
                collapsed = True
                logger.error(
                    "[driver] COLLAPSE at epoch %d: mean_R=%.3f is below %.0f%% of the best (%.3f at epoch %d). Stopping and rolling back to that snapshot; further epochs cannot recover because a group of all-zero rewards produces zero advantage.",
                    epoch + 1,
                    r_now,
                    100 * self.cfg.collapse_ratio,
                    best_reward,
                    best_epoch,
                )
                break
            if self.cfg.save_every and (epoch + 1) % self.cfg.save_every == 0:
                self.vlm.save_adapter(str(self.out_dir / f"adapter_epoch{epoch + 1:03d}"))
        final = str(self.out_dir / "adapter_final")
        if collapsed and Path(best_path).exists():
            import shutil

            shutil.rmtree(final, ignore_errors=True)
            shutil.copytree(best_path, final)
            logger.info("[driver] adapter_final ← best snapshot (epoch %d)", best_epoch)
        else:
            self.vlm.save_adapter(final)
        delta = self.adapter_delta()
        logger.info("[driver] adapter after training: %s", delta)
        if delta["lora_B_max"] < 0.001:
            logger.warning(
                "[driver] the learned update is negligible (max|B|=%.2e). With lr=%.1e over ~%d optimiser steps the model barely moved; any downstream comparison against the base checkpoint is measuring noise, not this objective.",
                delta["lora_B_max"],
                self.cfg.learning_rate,
                sum((h.get("n_updates", 0) for h in self.history)),
            )
        summary = {
            "adapter_path": final,
            "collapsed": collapsed,
            "best_epoch": best_epoch,
            "best_mean_reward": round(best_reward, 4) if best_reward >= 0 else None,
            "total_elapsed_h": round((time.time() - t0) / 3600, 3),
            "history": self.history,
            "adapter_delta": delta,
            "final_kg_stats": self.store.stats(),
            "config": asdict(self.cfg),
        }
        with open(self.out_dir / "self_play_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        return summary

    def _solver_prompt(self, q) -> str:
        base = f"[Question]\n{q.question_text}\n\n"
        snippets = self._evidence_for(q)
        if not snippets:
            return base + "Please reason step by step and provide your final answer."
        body = "\n".join((f"[{i + 1}] {t}" for i, t in enumerate(snippets)))
        return f"You are given numbered evidence snippets drawn from scientific documents. Some are relevant and some are not.\n\n{body}\n\n{base}Reason step by step, citing the snippet numbers you use and naming the specific findings, figures or measurements they contain. Then give the final answer on a new line starting with 'ANSWER:'."

    def _evidence_for(self, q, n_distractors: int = 5) -> List[str]:
        path = getattr(q, "reasoning_path", None)
        if path is None or not getattr(path, "nodes", None):
            return []
        gold: List[str] = []
        for n in path.nodes:
            c = (n.content or "").strip()
            if not c:
                continue
            for j in range(0, min(len(c), 900), 300):
                piece = c[j : j + 300].strip()
                if piece:
                    gold.append(piece)
        if not gold:
            return []
        papers = {n.paper_id for n in path.nodes if n.paper_id}
        pool: List[str] = []
        for pid in list(papers)[:4]:
            if pid not in self._paper_node_cache:
                self._paper_node_cache[pid] = self.store.get_nodes_by_paper(pid)
            for n in self._paper_node_cache[pid]:
                c = (n.content or "").strip()
                if len(c) >= 60:
                    pool.append(c[:300])
        rng = random.Random(hash(q.question_id) & 4294967295)
        gold_keys = {g[:120] for g in gold}
        distractors = [c for c in pool if c[:120] not in gold_keys]
        rng.shuffle(distractors)
        snippets = gold + distractors[:n_distractors]
        rng.shuffle(snippets)
        return snippets

    def _run_epoch(self, epoch: int) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        questions = self._make_questions(epoch)
        stats["n_questions"] = len(questions)
        if not questions:
            logger.warning("[epoch %d] no questions produced", epoch + 1)
            return stats
        reqs = [
            GenRequest(text=self._solver_prompt(q), image_paths=q.image_paths, tag=q.question_id)
            for q in questions
        ]
        sampled = self.vlm.generate(
            reqs,
            n_samples=self.cfg.n_samples_per_question,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
        )
        bundles = self._build_bundles(questions, sampled)
        bundles = self.reward.calculate(bundles, self.store)
        stats.update(self._reward_stats(bundles))
        if getattr(self.reward, "last_stats", None):
            stats["reward_diagnostics"] = self.reward.last_stats
        stats.update(self._grpo_update(bundles, epoch))
        if self.cfg.kg_refine:
            try:
                stats.update(self.refiner.refine(self.store, bundles, self.cfg.reward_threshold))
            except Exception as e:
                logger.warning("[driver] KG refinement failed: %s", e)
        if self.cfg.edge_feedback and getattr(self.reward, "edge_type_accuracy", None):
            self.q_gen.update_edge_accuracy(self.reward.edge_type_accuracy)
        self._save_qa_log(bundles, epoch)
        return stats

    def _make_questions(self, epoch: int):
        if self.question_source is not None:
            return self.question_source(self.vlm, epoch, self.cfg)
        paths = self._sample_paths(epoch)
        if not paths:
            return []
        prompts: List[Tuple[Any, GenRequest]] = []
        for p in paths:
            built = self._proposer_prompt(p)
            if built is not None:
                prompts.append((p, built))
        if not prompts:
            return []
        outs = self.vlm.generate(
            [r for _, r in prompts],
            n_samples=1,
            max_new_tokens=self.cfg.question_max_new_tokens,
            do_sample=True,
        )
        from spark_vlm.self_play.question_generator import GeneratedQuestion, EDGE_TO_QTYPE

        questions = []
        for i, (path, _) in enumerate(prompts):
            text = (outs[i][0] if outs[i] else "").strip()
            if not text or len(text) < 12:
                continue
            primary = path.edge_types[0] if path.edge_types else "CONTAINS"
            questions.append(
                GeneratedQuestion(
                    question_id=f"e{epoch:03d}_q{i:05d}",
                    question_text=text,
                    question_type=EDGE_TO_QTYPE.get(primary, "VQA"),
                    difficulty=path.difficulty,
                    context_text=self.q_gen._build_context(path),
                    image_paths=self.q_gen._collect_images(path),
                    reasoning_path=path,
                    gold_answer=path.gold_answer,
                    metadata={
                        "paper_ids": list({n.paper_id for n in path.nodes}),
                        "edge_types": path.edge_types,
                        "primary_edge": primary,
                    },
                )
            )
        return questions

    def _sample_paths(self, epoch: int):
        want = self.cfg.questions_per_epoch
        paths = []
        attempts = 0
        while len(paths) < want and attempts < 6:
            batch = self.q_gen.sample_reasoning_paths(
                store=self.store,
                paper_id=None,
                n_paths=int(want * 1.8) + 16,
                epoch=epoch,
                total_epochs=self.cfg.n_epochs,
            )
            for p in batch:
                if not p.nodes:
                    continue
                if self._allowed and any((n.paper_id not in self._allowed for n in p.nodes)):
                    continue
                paths.append(p)
                if len(paths) >= want:
                    break
            attempts += 1
        return paths[:want]

    def _proposer_prompt(self, path) -> Optional[GenRequest]:
        from spark_vlm.self_play.question_generator import _EDGE_PROMPTS, _SINGLE_NODE_PROMPT
        from spark_vlm.knowledge_graph.schema import NodeType

        images = self.q_gen._collect_images(path)
        hint = f"[Image available: {images[0]}]" if images else ""
        if any((n.node_type == NodeType.FIGURE for n in path.nodes)) and (not images):
            return None
        if not path.edge_types:
            node = path.nodes[0]
            text = _SINGLE_NODE_PROMPT.format(
                node_type=node.node_type.value,
                content=(node.content or "")[:600],
                image_hint=hint,
                gold_answer=path.gold_answer,
            )
        else:
            primary = path.edge_types[0]
            src, dst = (path.nodes[0], path.nodes[-1])
            tmpl = _EDGE_PROMPTS.get(primary, _EDGE_PROMPTS["DEFAULT"])
            text = tmpl.format(
                src_type=src.node_type.value,
                dst_type=dst.node_type.value,
                src_content=(src.content or "")[:400],
                dst_content=(dst.content or "")[:400],
                edge_type=primary,
                difficulty=path.difficulty,
                image_hint=hint,
                gold_answer=path.gold_answer,
            )
        return GenRequest(text=text, image_paths=images[:1])

    def _build_bundles(self, questions, sampled):
        from spark_vlm.self_play.answer_generator import AnswerBundle, GeneratedAnswer

        bundles = []
        for q, samples in zip(questions, sampled):
            answers = [
                GeneratedAnswer(
                    question_id=q.question_id,
                    answer_text=t,
                    sample_idx=i,
                    used_images=q.image_paths,
                )
                for i, t in enumerate(samples)
                if t and t.strip()
            ]
            if answers:
                bundles.append(AnswerBundle(question=q, answers=answers))
        return bundles

    @staticmethod
    def _reward_stats(bundles) -> Dict[str, Any]:
        tot, acc, pth, cons = ([], [], [], [])
        hops, per_group_pass = ([], [])
        edge_acc: Dict[str, List[float]] = {}
        for b in bundles:
            path = b.question.reasoning_path
            hops.append(max(0, len(path.node_ids or []) - 1))
            group_corr = []
            for a in b.answers:
                r = a.metadata.get("reward")
                if r is None:
                    continue
                tot.append(r.total)
                acc.append(r.r_correctness)
                pth.append(r.r_path)
                cons.append(r.r_consistency)
                group_corr.append(r.r_correctness)
            if group_corr:
                per_group_pass.append(sum((1 for c in group_corr if c >= 0.5)) / len(group_corr))
                primary = (path.edge_types or [""])[0]
                edge_acc.setdefault(primary, []).extend(group_corr)

        def _m(v):
            return round(sum(v) / len(v), 4) if v else None

        return {
            "n_groups": len(bundles),
            "n_scored_answers": len(tot),
            "mean_total_reward": _m(tot),
            "mean_r_correctness": _m(acc),
            "mean_r_path": _m(pth),
            "mean_r_consistency": _m(cons),
            "mean_hops": _m(hops),
            "hop_distribution": dict(sorted(Counter(hops).items())),
            "solver_pass_rate": _m(per_group_pass),
            "edge_type_accuracy": {k: _m(v) for k, v in sorted(edge_acc.items())},
        }

    def _grpo_update(self, bundles, epoch: int) -> Dict[str, Any]:
        import torch
        from torch.optim import AdamW

        model = self.vlm.model
        tok = self.vlm.text_tokenizer()
        if self._optimizer is None:
            params = [p for p in model.parameters() if p.requires_grad]
            self._optimizer = AdamW(params, lr=self.cfg.learning_rate)
            from torch.optim.lr_scheduler import LambdaLR

            total = max(
                1, self.cfg.n_epochs * max(1, self.cfg.questions_per_epoch // self.cfg.grad_accum)
            )
            warm = max(1, int(total * self.cfg.warmup_frac))

            def _sched(step: int) -> float:
                if step < warm:
                    return (step + 1) / warm
                prog = (step - warm) / max(1, total - warm)
                return max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

            self._lr_sched = LambdaLR(self._optimizer, _sched)
        opt = self._optimizer
        groups = [b for b in bundles if len(b.answers) >= 2]
        if self.cfg.max_groups_per_epoch:
            groups = groups[: self.cfg.max_groups_per_epoch]
        losses: List[float] = []
        n_updates = 0
        n_skipped = 0
        adv_abs: List[float] = []
        torch.cuda.empty_cache()
        model.train()
        opt.zero_grad(set_to_none=True)
        for gi, b in enumerate(groups):
            rewards = [
                a.metadata.get("reward").total if a.metadata.get("reward") else 0.0
                for a in b.answers
            ]
            n = len(rewards)
            mean_r = sum(rewards) / n
            var = sum(((r - mean_r) ** 2 for r in rewards)) / n
            std = math.sqrt(var)
            if std < 1e-06:
                n_skipped += 1
                continue
            advantages = [(r - mean_r) / (std + 1e-08) for r in rewards]
            adv_abs.extend((abs(a) for a in advantages))
            prompt = self._solver_prompt(b.question)
            if self.cfg.objective == "sft":
                best = max(
                    b.answers,
                    key=lambda a: (
                        a.metadata.get("reward").total if a.metadata.get("reward") else 0.0
                    ),
                )
                br = best.metadata.get("reward")
                if not br or br.total < self.cfg.reward_threshold:
                    continue
                group_loss = self._group_backward(model, tok, prompt, [best], [None])
            else:
                group_loss = self._group_backward(model, tok, prompt, b.answers, advantages)
            if group_loss is None:
                continue
            losses.append(group_loss)
            if (gi + 1) % self.cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], self.cfg.max_grad_norm
                )
                opt.step()
                if getattr(self, "_lr_sched", None) is not None:
                    self._lr_sched.step()
                opt.zero_grad(set_to_none=True)
                n_updates += 1
        if losses and len(groups) % self.cfg.grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], self.cfg.max_grad_norm
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            n_updates += 1
        model.eval()
        return {
            "n_updates": n_updates,
            "n_skipped_zero_variance": n_skipped,
            "mean_loss": round(sum(losses) / len(losses), 5) if losses else None,
            "mean_abs_advantage": round(sum(adv_abs) / len(adv_abs), 4) if adv_abs else None,
        }

    def _group_backward(self, model, tok, prompt: str, answers, advantages):
        import torch

        total = 0.0
        n_done = 0
        denom = len(answers) * self.cfg.grad_accum
        micro = max(1, self.cfg.train_micro_batch)
        for start in range(0, len(answers), micro):
            chunk = answers[start : start + micro]
            chunk_adv = advantages[start : start + micro]
            for ans, adv in zip(chunk, chunk_adv):
                enc_p = tok(prompt, return_tensors="pt", truncation=True, max_length=768)
                enc_a = tok(
                    ans.answer_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.cfg.max_new_tokens,
                    add_special_tokens=False,
                )
                if enc_a["input_ids"].shape[1] < 2:
                    continue
                dev = self.vlm.input_device
                ids = torch.cat([enc_p["input_ids"], enc_a["input_ids"]], dim=1).to(dev)
                mask = torch.ones_like(ids)
                labels = ids.clone()
                labels[:, : enc_p["input_ids"].shape[1]] = -100
                out = self.vlm.lm_forward(ids, mask, labels)
                logp = -out.loss
                kl = 0.0
                if adv is not None and self.cfg.kl_coeff:
                    with torch.no_grad():
                        with self.vlm.disable_adapter():
                            ref = self.vlm.lm_forward(ids, mask, labels)
                            ref_logp = -ref.loss.detach()
                    kl = logp - ref_logp
                if adv is None:
                    term = out.loss
                else:
                    a_t = torch.as_tensor(adv, device=logp.device, dtype=logp.dtype)
                    term = -a_t * logp + self.cfg.kl_coeff * kl
                (term / denom).backward()
                total += float(term.detach())
                n_done += 1
                del out, logp, kl, term
        return total / n_done if n_done else None

    def adapter_delta(self) -> Dict[str, float]:
        import torch

        mx = mean = 0.0
        n = 0
        for name, prm in self.vlm.model.named_parameters():
            if "lora_B" in name:
                a = prm.detach().abs()
                mx = max(mx, float(a.max()))
                mean += float(a.mean())
                n += 1
        return {
            "lora_B_max": round(mx, 8),
            "lora_B_mean": round(mean / max(n, 1), 8),
            "n_lora_B": n,
        }

    def _save_qa_log(self, bundles, epoch: int) -> None:
        d = self.out_dir / "qa_logs"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"epoch{epoch + 1:03d}_qa.jsonl", "w", encoding="utf-8") as f:
            for b in bundles:
                q = b.question
                f.write(
                    json.dumps(
                        {
                            "question_id": q.question_id,
                            "question_text": q.question_text,
                            "question_type": q.question_type,
                            "difficulty": q.difficulty,
                            "gold_answer": q.gold_answer,
                            "gold_path_node_ids": list(q.reasoning_path.node_ids or []),
                            "edge_types": q.metadata.get("edge_types", []),
                            "primary_edge": q.metadata.get("primary_edge", ""),
                            "paper_ids": q.metadata.get("paper_ids", []),
                            "image_paths": q.image_paths,
                            "answers": [
                                {
                                    "sample_idx": a.sample_idx,
                                    "answer_text": a.answer_text,
                                    "reward": _reward_row(a),
                                }
                                for a in b.answers
                            ],
                            "best_answer": b.best_answer.answer_text if b.best_answer else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def _reward_row(a) -> Dict[str, Any]:
    r = a.metadata.get("reward")
    if r is None:
        return {}
    row = {
        "total": r.total,
        "r_correctness": r.r_correctness,
        "r_path": r.r_path,
        "r_consistency": r.r_consistency,
        "r_challenger": r.r_challenger,
    }
    for k in ("p_solver", "unlinked_rate", "path_match_at_gold", "r_path_source"):
        if k in (r.metadata or {}):
            row[k] = r.metadata[k]
    return row
