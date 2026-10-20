import argparse
import json
import logging
from dataclasses import fields
from pathlib import Path

from spark_vlm.knowledge_graph.graph_store import NetworkXGraphStore
from spark_vlm.knowledge_graph.schema import KGEdge, KGNode


def read_graph(path, paper_ids=None):
    path = Path(path).resolve()
    store = NetworkXGraphStore()
    store.load(str(path))
    data = json.loads(path.read_text())
    node_ids = [raw["node_id"] for raw in data["nodes"]]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Node IDs must be unique")
    for raw in data["nodes"]:
        raw = dict(raw)
        raw.pop("id", None)
        node = KGNode.from_dict(raw)
        if node.image_path:
            image = Path(node.image_path)
            image = image if image.is_absolute() else path.parent / image
            if not image.is_file():
                raise ValueError(f"Missing image: {image}")
            node.image_path = str(image.resolve())
            store.add_node(node)
        if paper_ids is not None:
            if node.paper_id not in paper_ids or node.metadata.get("split") != "train":
                raise ValueError(f"Graph contains a non-training node: {node.node_id}")
    for raw in data["edges"]:
        edge = KGEdge.from_dict(dict(raw))
        if edge.src_id not in node_ids or edge.dst_id not in node_ids:
            raise ValueError("Edge references a missing node")
    if not data["nodes"]:
        raise ValueError("Graph has no nodes")
    return store


def train(args):
    from transformers import set_seed
    from spark_vlm.experiments.self_play_driver import SelfPlayConfig, SelfPlayDriver

    config = json.loads(Path(args.config).read_text())
    unknown = set(config) - {f.name for f in fields(SelfPlayConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    cfg = SelfPlayConfig(**config)
    if cfg.device in {"auto", "multi"}:
        raise ValueError("Set one explicit device, such as cuda:0")
    if cfg.objective not in {"spark", "sft"}:
        raise ValueError("objective must be spark or sft")
    if cfg.n_epochs < 1 or cfg.questions_per_epoch < 1 or cfg.n_samples_per_question < 2:
        raise ValueError("Use positive epochs/questions and at least two answers per question")
    if cfg.grad_accum < 1 or cfg.learning_rate <= 0:
        raise ValueError("grad_accum and learning_rate must be positive")
    papers = json.loads(Path(args.papers).read_text())
    if not isinstance(papers, list) or not papers or not all(isinstance(x, str) for x in papers):
        raise ValueError("papers must be a nonempty JSON list of training paper IDs")
    store = read_graph(args.kg, set(papers))
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory")
    set_seed(cfg.seed)
    summary = SelfPlayDriver(cfg, store, output, paper_ids=papers).run()
    print(json.dumps(summary, indent=2))


def predict(args):
    from transformers import set_seed
    from spark_vlm.experiments.batched_vlm import BatchedVLM, GenRequest

    source = Path(args.input).resolve()
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Input IDs must be unique")
    requests = []
    for row in rows:
        images = []
        for image in row.get("images", []):
            path = Path(image)
            path = path if path.is_absolute() else source.parent / path
            if not path.is_file():
                raise ValueError(f"Missing image: {path}")
            images.append(str(path.resolve()))
        if len(images) > 2:
            raise ValueError("This backend supports at most two images per request")
        requests.append(GenRequest(row["question"], images, str(row["id"])))
    set_seed(args.seed)
    backend = BatchedVLM(
        args.model, device=args.device, adapter_path=args.adapter, trainable=False, gen_batch_size=1
    )
    with Path(args.output).open("x") as stream:
        for row, request in zip(rows, requests):
            response = backend.generate(
                [request], n_samples=1, max_new_tokens=args.max_new_tokens, do_sample=False
            )[0][0]
            stream.write(
                json.dumps({"id": row["id"], "response": response}, ensure_ascii=False) + "\n"
            )
            stream.flush()


def main():
    parser = argparse.ArgumentParser(prog="spark")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-graph")
    check.add_argument("--kg", required=True)
    run = commands.add_parser("train")
    for option in ["kg", "papers", "config", "output"]:
        run.add_argument("--" + option, required=True)
    pred = commands.add_parser("predict")
    for option in ["model", "input", "output"]:
        pred.add_argument("--" + option, required=True)
    pred.add_argument("--adapter")
    pred.add_argument("--device", default="cuda:0")
    pred.add_argument("--max-new-tokens", type=int, default=2048)
    pred.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "check-graph":
        print(json.dumps(read_graph(args.kg).stats(), indent=2))
    elif args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
