"""Root-consistent rational scaling and PCG64 token synthesis."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .trace_loader import TraceRequest, sha256_file


@dataclass(frozen=True)
class Scale:
    numerator: int
    denominator: int
    budget: int

    @classmethod
    def for_root(cls, requests: list[TraceRequest], budget: int):
        if budget < 2:
            raise ValueError("context_budget must be at least 2")
        peak = max(r.input_len + r.output_len for r in requests)
        return cls(min(budget, peak), peak, budget)

    def floor(self, tokens: int) -> int:
        return tokens * self.numerator // self.denominator

    def ceil(self, tokens: int) -> int:
        return (tokens * self.numerator + self.denominator - 1) // self.denominator

    def q(self, block_index: int) -> int:
        return self.floor(64 * block_index)


def common_prefix(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    if not n:
        return 0
    different = a[:n] != b[:n]
    first = int(different.argmax())
    return first if different[first] else n


def check_positions(requests: list[TraceRequest]) -> int:
    positions = {}
    for request in requests:
        for position, value in enumerate(request.hash_ids):
            key = (request.root_id, int(value))
            previous = positions.setdefault(key, position)
            if previous != position:
                raise ValueError(
                    f"Hash position conflict: {key}: {previous} vs {position}"
                )
    return len(positions)


def tokenizer_vocabulary(path: Path) -> tuple[np.ndarray, dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    special = set(tokenizer.all_special_ids)
    special.update(
        i for i, token in tokenizer.added_tokens_decoder.items() if token.special
    )
    actual = set(tokenizer.get_vocab().values())
    eligible = np.asarray(sorted(actual - special), dtype=np.int32)
    if not len(eligible):
        raise ValueError("Empty non-special vocabulary")
    files = ["tokenizer.json", "tokenizer_config.json", "config.json"]
    return eligible, {
        "model_name": "Qwen/Qwen3-8B",
        "local_path": str(path.resolve()),
        "tokenizer_class": type(tokenizer).__name__,
        "vocabulary_size": len(actual),
        "eligible_size": len(eligible),
        "excluded_special_ids": sorted(special),
        "eligible_ids_sha256": hashlib.sha256(
            eligible.astype("<i4").tobytes()
        ).hexdigest(),
        "files": {
            name: sha256_file(path / name) for name in files if (path / name).exists()
        },
    }


class Synthesizer:
    def __init__(self, root_id: str, scale: Scale, vocabulary: np.ndarray):
        self.root_id = root_id
        self.scale = scale
        self.vocabulary = vocabulary
        self.blocks: dict[int, tuple[int, np.ndarray]] = {}

    def prompt(
        self,
        request: TraceRequest,
        previous: TraceRequest | None = None,
        previous_tokens: np.ndarray | None = None,
    ) -> np.ndarray:
        if request.root_id != self.root_id:
            raise ValueError("A synthesizer cannot span root namespaces")
        shared = 0
        if previous is not None:
            if previous.root_id != self.root_id or previous_tokens is None:
                raise ValueError("Invalid prefix reuse")
            shared = common_prefix(previous.hash_ids, request.hash_ids)
        prompt = np.empty(self.scale.floor(request.input_len), dtype=np.int32)
        if not len(prompt):
            raise ValueError("Scaling produced an empty prompt")
        if shared:
            prompt[: self.scale.q(shared)] = previous_tokens[: self.scale.q(shared)]
        for k in range(shared, len(request.hash_ids)):
            h = int(request.hash_ids[k])
            start, end = self.scale.q(k), self.scale.q(k + 1)
            if end <= start:
                raise ValueError(
                    "Scaling collapsed a source block; use a larger budget"
                )
            cached = self.blocks.get(h)
            if cached is None:
                digest = hashlib.sha256(
                    (self.root_id + str(h)).encode("utf-8")
                ).digest()
                seed = int.from_bytes(digest[:8], "big")
                rng = np.random.Generator(np.random.PCG64(seed))
                tokens = rng.choice(self.vocabulary, size=end - start, replace=True)
                self.blocks[h] = (k, tokens)
            else:
                position, tokens = cached
                if position != k:
                    raise ValueError(f"Hash position conflict: {self.root_id}/{h}")
            prompt[start:end] = tokens
        return prompt
