"""Train the trace safety classifier on ``processed_traces.json``.

Run from the repository root, for example:
    python -m src.models.train --epochs 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.models.classifier import TraceClassifier


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "src" / "data" / "processed_traces.json"
DEFAULT_CHECKPOINT_PATH = REPOSITORY_ROOT / "models" / "trace_classifier.pt"
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def token_to_id(token: str, vocab_size: int) -> int:
    """Map a token to a stable non-padding vocabulary id without a fitted vocab."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (vocab_size - 1) + 1


class TraceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Loads labelled traces and converts their text to fixed-size token-id tensors."""

    def __init__(self, data_path: Path, max_length: int, vocab_size: int) -> None:
        with data_path.open(encoding="utf-8") as data_file:
            records = json.load(data_file)

        if not isinstance(records, list) or not records:
            raise ValueError(f"{data_path} must contain a non-empty JSON list of records.")

        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict) or "text" not in record or "label" not in record:
                raise ValueError(f"Record {index} is missing a text or label field.")
            label = int(record["label"])
            if label not in (0, 1):
                raise ValueError(f"Record {index} has invalid binary label {label!r}.")

            ids = [token_to_id(token.lower(), vocab_size) for token in TOKEN_PATTERN.findall(str(record["text"]))]
            ids = ids[:max_length]
            ids.extend([0] * (max_length - len(ids)))
            self.examples.append((torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


def split_dataset(dataset: Dataset, validation_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if len(dataset) < 2:
        raise ValueError("At least two records are required for a train/validation split.")

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, round(len(indices) * validation_fraction))
    validation_size = min(validation_size, len(indices) - 1)
    return (
        torch.utils.data.Subset(dataset, indices[validation_size:]),
        torch.utils.data.Subset(dataset, indices[:validation_size]),
    )


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train(args: argparse.Namespace) -> TraceClassifier:
    torch.manual_seed(args.seed)
    data_path = Path(args.data).expanduser()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Generate it with python src/data/dataset.py.")

    dataset = TraceDataset(data_path, args.max_length, args.vocab_size)
    if args.limit is not None:
        dataset.examples = dataset.examples[: args.limit]
    train_set, validation_set = split_dataset(dataset, args.validation_fraction, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TraceClassifier(vocab_size=args.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for input_ids, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} | train loss: {train_loss / seen:.4f} | "
            f"validation loss: {validation_loss:.4f} | validation accuracy: {validation_accuracy:.2%}"
        )

    checkpoint_path = Path(args.output).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "vocab_size": args.vocab_size, "max_length": args.max_length},
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")
    return model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AgentGuard trace classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=30_522)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N records (for quick checks).")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1 or args.vocab_size < 2:
        parser.error("epochs, batch-size, max-length, and vocab-size must be positive (vocab-size >= 2).")
    if args.limit is not None and args.limit < 2:
        parser.error("limit must be at least 2.")
    return args


if __name__ == "__main__":
    train(parse_args())
