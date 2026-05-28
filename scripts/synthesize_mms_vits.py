from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

from mms_hau_pipeline import (
    VocabTextMapper,
    load_pronunciation_overrides,
    normalize_hausa_text,
    normalize_override_entries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize speech from a fine-tuned MMS Hausa VITS checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the finetune config JSON.")
    parser.add_argument("--vits-dir", required=True, help="Path to the cloned jaywalnut310/vits repository.")
    parser.add_argument("--checkpoint", default=None, help="Generator checkpoint path. Defaults to the latest G_*.pth in --checkpoint-dir.")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory containing saved generator checkpoints.")
    parser.add_argument("--text", required=True, help="Input text to synthesize.")
    parser.add_argument("--out-wav", required=True, help="Output WAV file path.")
    parser.add_argument("--pronunciation-overrides", default=None, help="Optional source|target override file.")
    parser.add_argument("--noise-scale", type=float, default=0.667, help="Sampling noise scale.")
    parser.add_argument("--noise-scale-w", type=float, default=0.8, help="Duration noise scale.")
    parser.add_argument("--length-scale", type=float, default=1.0, help="Speech speed control. >1.0 is slower.")
    return parser.parse_args()


def latest_generator_checkpoint(checkpoint_dir: Path) -> Path:
    matches = [Path(path) for path in glob.glob(str(checkpoint_dir / "G_*.pth"))]
    if not matches:
        raise FileNotFoundError(f"No generator checkpoints were found in {checkpoint_dir}")
    return max(matches, key=lambda path: int("".join(filter(str.isdigit, path.stem)) or 0))


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    vits_dir = Path(args.vits_dir).resolve()
    sys.path.insert(0, str(vits_dir))

    import models  # type: ignore
    import utils  # type: ignore

    text_mapper = VocabTextMapper(config["data"]["vocab_file"])
    overrides = normalize_override_entries(load_pronunciation_overrides(args.pronunciation_overrides), text_mapper.symbols)
    text = normalize_hausa_text(args.text, text_mapper.symbols, overrides)

    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else latest_generator_checkpoint(Path(args.checkpoint_dir).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_g = models.SynthesizerTrn(
        len(text_mapper.symbols),
        int(config["data"]["filter_length"]) // 2 + 1,
        int(config["train"]["segment_size"]) // int(config["data"]["hop_length"]),
        **config["model"],
    ).to(device)
    net_g.eval()
    utils.load_checkpoint(str(checkpoint_path), net_g, None)

    text_ids = torch.LongTensor(text_mapper.text_to_ids(text, add_blank=bool(config["data"]["add_blank"]))).unsqueeze(0).to(device)
    text_lengths = torch.LongTensor([text_ids.size(1)]).to(device)

    with torch.no_grad():
        waveform = net_g.infer(
            text_ids,
            text_lengths,
            noise_scale=args.noise_scale,
            noise_scale_w=args.noise_scale_w,
            length_scale=args.length_scale,
        )[0][0, 0].cpu().float().numpy()

    output_path = Path(args.out_wav).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(output_path, int(config["data"]["sampling_rate"]), waveform.astype(np.float32))
    print(f"Synthesized audio saved to {output_path}")


if __name__ == "__main__":
    main()
