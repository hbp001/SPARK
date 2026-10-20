import json
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from spark_vlm.cli import read_graph
from spark_vlm.experiments.self_play_driver import SelfPlayConfig, SelfPlayDriver
from spark_vlm.knowledge_graph.schema import EdgeType, KGEdge, KGNode, NodeType
from spark_vlm.self_play.answer_generator import AnswerBundle, GeneratedAnswer
from spark_vlm.self_play.question_generator import (
    GeneratedQuestion,
    QuestionGenerator,
    ReasoningPath,
)
from spark_vlm.self_play.reward_calculator import KGVerifier, RewardCalculator


EXAMPLE = Path(__file__).parents[1] / "examples/graph.json"


def test_graph_roundtrip_and_sampling(tmp_path):
    store = read_graph(EXAMPLE, {"demo"})
    path = tmp_path / "graph.json"
    store.save(str(path))
    restored = read_graph(path, {"demo"})
    assert restored.stats() == store.stats()
    random.seed(42)
    paths = QuestionGenerator({}, {}).sample_reasoning_paths(restored, n_paths=8)
    assert 0 < len(paths) <= 8
    assert all(n.paper_id == "demo" for p in paths for n in p.nodes)


@pytest.mark.parametrize("split,paper", [("test", "demo"), ("train", "foreign")])
def test_nontraining_graph_is_rejected(tmp_path, split, paper):
    data = json.loads(EXAMPLE.read_text())
    data["nodes"][0]["metadata"]["split"] = split
    data["nodes"][0]["paper_id"] = paper
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="non-training"):
        read_graph(path, {"demo"})


def test_missing_edge_endpoint_is_rejected(tmp_path):
    data = json.loads(EXAMPLE.read_text())
    data["edges"][0]["dst_id"] = "absent"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing node"):
        read_graph(path)


def test_reward_uses_graph_answer():
    store = read_graph(EXAMPLE)
    node = store.get_node("demo_concept")
    path = ReasoningPath(["demo_concept"], [], [node], 1, node.content)
    question = GeneratedQuestion(
        "q1", "What does the sensor measure?", "VQA", 1, node.content, [], path, node.content
    )
    good = GeneratedAnswer("q1", node.content, 0, [])
    bad = GeneratedAnswer("q1", "unrelated bananas", 1, [])
    bundle = AnswerBundle(question, [good, bad])
    RewardCalculator({}).calculate([bundle], store)
    assert good.metadata["reward"].r_correctness > bad.metadata["reward"].r_correctness
    assert KGVerifier(store).extract_gold_answer(bundle) == node.content


def make_cpu_driver():
    import torch

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(6))

        def forward(self, ids, mask, labels):
            active = labels[:, 1:] != -100
            targets = labels[:, 1:][active]
            scores = self.logits.expand(targets.numel(), -1)
            return SimpleNamespace(loss=torch.nn.functional.cross_entropy(scores, targets))

    def tokenizer(text, **kwargs):
        ids = [1, 2] if text == "prompt" else [3, 3] if text == "good" else [4, 4]
        return {"input_ids": torch.tensor([ids])}

    policy = TinyPolicy()
    driver = object.__new__(SelfPlayDriver)
    driver.cfg = SelfPlayConfig(
        grad_accum=1,
        train_micro_batch=1,
        kl_coeff=0,
        learning_rate=0.01,
        n_epochs=1,
        questions_per_epoch=1,
    )
    driver.vlm = SimpleNamespace(
        model=policy,
        input_device="cpu",
        lm_forward=policy.forward,
        text_tokenizer=lambda: tokenizer,
        disable_adapter=lambda: nullcontext(),
    )
    driver._optimizer = None
    driver._solver_prompt = lambda question: "prompt"
    return driver, policy


@pytest.mark.parametrize("rewards,updates", [([1.0, 0.0], 1), ([1.0, 1.0], 0)])
def test_group_relative_gradient_updates(rewards, updates):
    import torch

    driver, policy = make_cpu_driver()
    before = policy.logits.detach().clone()
    answers = [
        GeneratedAnswer("q", text, i, [], {"reward": SimpleNamespace(total=reward)})
        for i, (text, reward) in enumerate(zip(["good", "bad"], rewards))
    ]
    stats = driver._grpo_update([AnswerBundle(None, answers)], epoch=0)
    assert stats["n_updates"] == updates
    assert torch.equal(before, policy.logits.detach()) == (updates == 0)
    if updates:
        assert policy.logits[3] > policy.logits[4]


def test_schema_roundtrip():
    node = KGNode("n", NodeType.CONCEPT, "demo", content="sensor")
    edge = KGEdge("n", "m", EdgeType.DEFINES)
    assert KGNode.from_dict(node.to_dict()) == node
    assert KGEdge.from_dict(edge.to_dict()) == edge
