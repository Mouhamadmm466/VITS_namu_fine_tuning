# MMS Hausa VITS Fine-Tuning Pipeline

This repo prepares a single-speaker Hausa dataset and fine-tunes the `facebook/mms-tts-hau` checkpoint with the original VITS architecture.

Important detail: the Hugging Face `transformers` wrapper for `VitsModel` supports inference, but its current source still raises `NotImplementedError` for training. This pipeline therefore uses:

- Hugging Face MMS full checkpoint files as the pretrained starting point
- the original [jaywalnut310/vits](https://github.com/jaywalnut310/vits) modules for the actual fine-tuning loop

## What is in this repo

- `metadata_wav.csv` and `wav/`: your source dataset
- `configs/mms_hau_base_config.json`: the official MMS Hausa full-model config
- `configs/mms_hau_vocab.txt`: the official MMS Hausa vocab
- `configs/pronunciation_overrides.tsv`: optional pronunciation control file
- `scripts/prepare_mms_vits_dataset.py`: validates data, normalizes text, resamples audio to 16 kHz, writes filelists and a Colab-ready config
- `scripts/finetune_mms_vits.py`: single-GPU fine-tuning script for Colab
- `scripts/synthesize_mms_vits.py`: inference script for the resulting checkpoints

## Why the text pipeline is custom

MMS Hausa is not using the stock English-style phonemizer flow from the original VITS repo. The released MMS inference code uses a direct character vocabulary and filters text against the checkpoint vocab.

This pipeline follows that behavior and adds an optional `source|target` override file so you can push pronunciation for specific words or phrases without changing the whole dataset.

## Local dataset preparation

Run this in the current repo before moving the prepared dataset to Colab:

```bash
python3 scripts/prepare_mms_vits_dataset.py \
  --metadata-csv metadata_wav.csv \
  --audio-root . \
  --output-dir prepared_mms_hau \
  --pronunciation-overrides configs/pronunciation_overrides.tsv
```

This will:

- resample all labeled WAV files to `16 kHz`
- normalize and filter text to the official MMS Hausa vocab
- split train and validation sets deterministically
- write `prepared_mms_hau/finetune_config.json`

Note: `wav/audio_015 2.wav` is not referenced by `metadata_wav.csv`, so it is intentionally ignored.

## Colab setup

After uploading or mounting this repo in Colab, run:

```bash
git clone https://github.com/jaywalnut310/vits.git /content/vits
pip install soundfile scipy tensorboard cython
cd /content/vits/monotonic_align
python setup.py build_ext --inplace
```

Download the MMS Hausa full checkpoint files:

```bash
mkdir -p /content/mms_hau_full
wget -O /content/mms_hau_full/G_100000.pth https://huggingface.co/facebook/mms-tts/resolve/main/full_models/hau/G_100000.pth
wget -O /content/mms_hau_full/D_100000.pth https://huggingface.co/facebook/mms-tts/resolve/main/full_models/hau/D_100000.pth
wget -O /content/mms_hau_full/config.json https://huggingface.co/facebook/mms-tts/resolve/main/full_models/hau/config.json
wget -O /content/mms_hau_full/vocab.txt https://huggingface.co/facebook/mms-tts/resolve/main/full_models/hau/vocab.txt
```

## Fine-tuning in Colab

Assuming this repo is available at `/content/Namu_tts`:

```bash
cd /content/Namu_tts
python3 scripts/finetune_mms_vits.py \
  --config /content/Namu_tts/prepared_mms_hau/finetune_config.json \
  --vits-dir /content/vits \
  --pretrained-dir /content/mms_hau_full
```

Checkpoints and TensorBoard logs will be written to the `train.output_dir` stored in the generated config. By default that is:

```text
prepared_mms_hau/runs/mms_hau_single_speaker
```

## Synthesis after fine-tuning

```bash
python3 scripts/synthesize_mms_vits.py \
  --config /content/Namu_tts/prepared_mms_hau/finetune_config.json \
  --vits-dir /content/vits \
  --checkpoint-dir /content/Namu_tts/prepared_mms_hau/runs/mms_hau_single_speaker \
  --text "sannu yaya kake yau" \
  --out-wav /content/final_sample.wav \
  --pronunciation-overrides /content/Namu_tts/configs/pronunciation_overrides.tsv
```

## Practical notes for this dataset

- Your current dataset is very small: 30 clips and about 1.38 minutes total.
- This is enough for a lightweight speaker adaptation attempt, but not enough for a stable, highly natural voice clone.
- The strongest lever you currently have for pronunciation is transcript quality and targeted override rules.
- Keep text lowercase and avoid punctuation that is not in the MMS vocab unless you intentionally map it through overrides.

## Recommended next improvements

- Add more speaker audio if possible, ideally clean recordings with broader phonetic coverage.
- Add targeted pronunciation overrides for recurring names, loanwords, or words the base model says poorly.
- Keep a fixed evaluation sentence list so you can compare checkpoints consistently.
