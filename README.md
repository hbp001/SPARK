# SPARK

Knowledge-graph-guided self-play for vision-language models.

SPARK samples reasoning paths from a document knowledge graph, generates questions and multiple answers, scores each answer against graph evidence, and updates a LoRA policy using group-relative rewards. The graph can be refined between epochs.

## Installation

Use Python 3.10 or newer. GPU training requires a compatible NVIDIA CUDA installation.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train]'
```

The optional training environment pins PyTorch 2.7.0, Transformers 4.57.1 and PEFT 0.20.0. Model repositories may require additional dependencies. Review remote model code before running InternVL chat checkpoints, which use `trust_remote_code=True`.

## Training

Prepare a graph JSON and a JSON list of training paper IDs. Image paths in the graph are resolved relative to the graph file. Every graph node must belong to an explicitly selected training paper and have `metadata.split` set to `train`.

```bash
spark check-graph --kg examples/graph.json
CUDA_VISIBLE_DEVICES=0 spark train \
  --kg examples/graph.json \
  --papers examples/train_papers.json \
  --config configs/qwen3vl4b.json \
  --output outputs/example
```

The example graph contains synthetic text for checking the data interface. Replace it with your training graph and locally available images for actual use. Model weights are downloaded by Hugging Face when needed; no weights or research datasets are distributed here.

Set `model_name` to a Hugging Face model ID or local model directory. `objective: "sft"` enables the original matched-data SFT control. Other settings are defined by `SelfPlayConfig` in `src/spark_vlm/experiments/self_play_driver.py`. Use an empty output directory for each run and preserve all seeds.

The output includes adapter weights, per-epoch QA logs, a training summary and graph statistics. Reward-based collapse handling is part of the original driver. This example configuration is a starting point, not a recipe for reproducing a particular published score.

## Prediction and evaluation

```bash
CUDA_VISIBLE_DEVICES=0 spark predict \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --adapter outputs/example/adapter_final \
  --input examples/questions.jsonl \
  --output predictions.jsonl \
  --max-new-tokens 2048
```

Omit `--adapter` for the same-model baseline. Input records contain `id`, `question`, and an optional `images` list. Generation is greedy with a fixed seed. For benchmarks, download the data from its original provider, construct questions with the official prompt, and score saved responses with that benchmark's official evaluator. Keep the model, processor, data, prompt and token limit identical between baseline and adapter runs.

## Implementation

```text
src/spark_vlm/
  cli.py
  experiments/       self-play driver and model backends
  knowledge_graph/   graph schema and storage
  self_play/         questions, answers, rewards and graph refinement
```

This release preserves the original core computation while reorganizing imports and removing commentary. Its gradient pass is text-only; image-conditioned generation and image-conditioned optimization are distinct. The reference penalty is the original sampled log-probability difference, not an exact full-distribution KL calculation. These implementation details should be considered when designing experiments. The release does not claim that every historical checkpoint or benchmark score has been reproduced.

## Tests

```bash
pip install -e '.[dev]'
pytest
```

CPU tests cover graph round-trips, training-only boundaries, reward computation, zero-variance groups and actual gradient updates. They do not substitute for GPU training or benchmark reproduction.
