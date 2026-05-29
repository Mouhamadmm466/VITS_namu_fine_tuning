from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from mms_hau_pipeline import (
    build_finetune_config,
    load_pronunciation_overrides,
    load_vocab_symbols,
    normalize_hausa_text,
    normalize_override_entries,
)


DEFAULT_BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mms_hau_base_config.json"
DEFAULT_VOCAB = Path(__file__).resolve().parents[1] / "configs" / "mms_hau_vocab.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a single-speaker MMS Hausa VITS dataset.")
    parser.add_argument("--metadata-csv", default="metadata_wav.csv", help="CSV with audio_path and transcription.")
    parser.add_argument("--audio-root", default=".", help="Base directory used to resolve relative audio paths.")
    parser.add_argument("--output-dir", default="prepared_mms_hau", help="Directory for resampled audio and filelists.")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG), help="Base MMS full-model config JSON.")
    parser.add_argument("--vocab-path", default=str(DEFAULT_VOCAB), help="Vocabulary file for facebook/mms-tts-hau.")
    parser.add_argument(
        "--pronunciation-overrides",
        default=None,
        help="Optional TSV-like file with one 'source|target' replacement per line.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate. MMS Hausa uses 16000.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for deterministic splitting.")
    parser.add_argument("--batch-size", type=int, default=4, help="Suggested training batch size written to config.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Suggested learning rate written to config.")
    parser.add_argument("--max-steps", type=int, default=1500, help="Suggested max steps written to config.")
    return parser.parse_args()


def load_rows(metadata_csv: Path) -> list[dict[str, str]]:
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"audio_path", "transcription"}
    if not rows:
        raise ValueError(f"No rows were found in {metadata_csv}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{metadata_csv} is missing required columns: {sorted(missing)}. Found: {sorted(rows[0])}")
    return rows


def resample_audio(source_path: Path, target_path: Path, target_sample_rate: int) -> int:
    audio, sample_rate = sf.read(source_path, always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != target_sample_rate:
        audio = resample_poly(audio, target_sample_rate, sample_rate)

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target_path, audio, target_sample_rate, subtype="PCM_16")
    return audio.shape[0]


def main() -> None:
    args = parse_args()

    metadata_csv = Path(args.metadata_csv).resolve()
    audio_root = Path(args.audio_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    wav_dir = output_dir / "wavs"
    filelists_dir = output_dir / "filelists"
    filelists_dir.mkdir(parents=True, exist_ok=True)

    vocab_symbols = load_vocab_symbols(args.vocab_path)
    overrides = normalize_override_entries(load_pronunciation_overrides(args.pronunciation_overrides), vocab_symbols)
    rows = load_rows(metadata_csv)

    prepared_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        source_audio_path = Path(row["audio_path"])
        if not source_audio_path.is_absolute():
            source_audio_path = (audio_root / source_audio_path).resolve()
        if not source_audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {source_audio_path}")

        normalized_text = normalize_hausa_text(
            row["transcription"],
            allowed_symbols=vocab_symbols,
            overrides=overrides,
        )

        target_audio_path = wav_dir / f"{index:04d}.wav"
        num_samples = resample_audio(source_audio_path, target_audio_path, args.sample_rate)

        prepared_rows.append(
            {
                "audio_path": str(target_audio_path),
                "text": normalized_text,
                "duration_seconds": num_samples / args.sample_rate,
                "source_audio_path": str(source_audio_path),
            }
        )

    random.Random(args.seed).shuffle(prepared_rows)
    val_count = max(1, round(len(prepared_rows) * args.val_ratio)) if len(prepared_rows) > 1 else 0
    val_rows = prepared_rows[:val_count]
    train_rows = prepared_rows[val_count:] if val_count else prepared_rows

    if not train_rows:
        raise ValueError("Training split is empty. Reduce --val-ratio.")

    def write_filelist(path: Path, items: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(f"{item['audio_path']}|{item['text']}\n")

    train_filelist = filelists_dir / "train.txt.cleaned"
    val_filelist = filelists_dir / "val.txt.cleaned"
    all_filelist = filelists_dir / "all.txt.cleaned"

    write_filelist(train_filelist, train_rows)
    write_filelist(val_filelist, val_rows)
    write_filelist(all_filelist, prepared_rows)

    config = build_finetune_config(
        base_config_path=args.base_config,
        train_filelist=train_filelist,
        val_filelist=val_filelist,
        vocab_path=args.vocab_path,
        output_dir=output_dir / "runs" / "mms_hau_single_speaker",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    config_path = output_dir / "finetune_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "num_samples": len(prepared_rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "sample_rate": args.sample_rate,
        "metadata_csv": str(metadata_csv),
        "audio_root": str(audio_root),
        "vocab_path": str(Path(args.vocab_path).resolve()),
        "pronunciation_overrides": str(Path(args.pronunciation_overrides).resolve()) if args.pronunciation_overrides else None,
        "rows": prepared_rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    total_minutes = sum(float(item["duration_seconds"]) for item in prepared_rows) / 60.0
    print(f"Prepared {len(prepared_rows)} clips at {args.sample_rate} Hz.")
    print(f"Train clips: {len(train_rows)} | Validation clips: {len(val_rows)}")
    print(f"Total audio: {total_minutes:.2f} minutes")
    print(f"Wrote config to: {config_path}")


if __name__ == "__main__":
    main()
