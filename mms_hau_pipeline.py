from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable


def load_vocab_symbols(vocab_path: str | Path) -> list[str]:
    vocab_file = Path(vocab_path)
    symbols = [line.rstrip("\n") for line in vocab_file.read_text(encoding="utf-8").splitlines()]
    if not symbols:
        raise ValueError(f"Vocabulary file is empty: {vocab_file}")
    return symbols


def load_pronunciation_overrides(path: str | Path | None) -> list[tuple[str, str]]:
    if path is None:
        return []

    override_path = Path(path)
    if not override_path.exists():
        raise FileNotFoundError(f"Pronunciation override file was not found: {override_path}")

    overrides: list[tuple[str, str]] = []
    for raw_line in override_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                "Each pronunciation override line must use 'source|target' format. "
                f"Problematic line: {raw_line!r}"
            )
        overrides.append((parts[0], parts[1]))
    return overrides


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def apply_phrase_overrides(text: str, overrides: Iterable[tuple[str, str]]) -> str:
    result = text
    for source, target in sorted(overrides, key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(rf"(?<!\S){re.escape(source)}(?!\S)")
        result = pattern.sub(target, result)
    return result


def normalize_hausa_text(
    text: str,
    allowed_symbols: Iterable[str],
    overrides: Iterable[tuple[str, str]] | None = None,
) -> str:
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    cleaned = unicodedata.normalize("NFKC", text).strip().lower()
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = cleaned.replace("’", "'").replace("`", "'").replace("´", "'")
    cleaned = cleaned.replace("“", "").replace("”", "").replace('"', "")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"[.,!?;:()\\[\\]{}]", " ", cleaned)
    cleaned = collapse_spaces(cleaned)

    if overrides:
        cleaned = apply_phrase_overrides(cleaned, overrides)
        cleaned = collapse_spaces(cleaned)

    allowed = set(allowed_symbols)
    filtered = "".join(char for char in cleaned if char in allowed)
    filtered = collapse_spaces(filtered)

    if not filtered:
        raise ValueError(f"Text became empty after normalization: {text!r}")

    return filtered


def normalize_override_entries(
    overrides: Iterable[tuple[str, str]],
    allowed_symbols: Iterable[str],
) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for source, target in overrides:
        normalized.append(
            (
                normalize_hausa_text(source, allowed_symbols, overrides=None),
                normalize_hausa_text(target, allowed_symbols, overrides=None),
            )
        )
    return normalized


class VocabTextMapper:
    def __init__(self, vocab_path: str | Path):
        self.vocab_path = Path(vocab_path)
        self.symbols = load_vocab_symbols(self.vocab_path)
        self.symbol_to_id = {symbol: index for index, symbol in enumerate(self.symbols)}
        self.id_to_symbol = {index: symbol for index, symbol in enumerate(self.symbols)}

    def text_to_ids(self, text: str, add_blank: bool = True) -> list[int]:
        ids = [self.symbol_to_id[symbol] for symbol in text]
        if add_blank:
            ids = intersperse(ids, 0)
        return ids

    def ids_to_text(self, ids: Iterable[int]) -> str:
        return "".join(self.id_to_symbol[index] for index in ids)


def intersperse(values: list[int], blank_id: int) -> list[int]:
    result = [blank_id] * (len(values) * 2 + 1)
    result[1::2] = values
    return result


def build_finetune_config(
    base_config_path: str | Path,
    train_filelist: str | Path,
    val_filelist: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    max_steps: int = 1500,
    log_interval: int = 10,
    eval_interval: int = 100,
    save_interval: int = 100,
    num_workers: int = 2,
    seed: int = 1234,
) -> dict:
    config = json.loads(Path(base_config_path).read_text(encoding="utf-8"))
    config["data"]["training_files"] = str(Path(train_filelist).resolve())
    config["data"]["validation_files"] = str(Path(val_filelist).resolve())
    config["data"]["vocab_file"] = str(Path(vocab_path).resolve())
    config["train"]["batch_size"] = batch_size
    config["train"]["learning_rate"] = learning_rate
    config["train"]["seed"] = seed
    config["train"]["log_interval"] = log_interval
    config["train"]["eval_interval"] = eval_interval
    config["train"]["save_interval"] = save_interval
    config["train"]["max_steps"] = max_steps
    config["train"]["num_workers"] = num_workers
    config["train"]["output_dir"] = str(Path(output_dir).resolve())
    return config
