from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import glob
import itertools
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from mms_hau_pipeline import VocabTextMapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU MMS Hausa VITS fine-tuning.")
    parser.add_argument("--config", required=True, help="Path to the generated finetune config JSON.")
    parser.add_argument("--vits-dir", required=True, help="Path to the cloned jaywalnut310/vits repository.")
    parser.add_argument(
        "--pretrained-dir",
        required=True,
        help="Directory containing the MMS full checkpoint files G_100000.pth and D_100000.pth.",
    )
    parser.add_argument("--output-dir", default=None, help="Override train.output_dir from the config.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument(
        "--trainable-modules",
        default="dec,dp",
        help=(
            "Comma-separated list of generator module name prefixes to train. "
            "All other parameters are frozen. "
            "Use 'dec' for decoder-only (voice timbre), 'dec,dp' to also adapt the duration predictor. "
            "Pass 'all' to fine-tune the entire generator (only safe with large datasets)."
        ),
    )
    return parser.parse_args()


def freeze_generator(net_g, trainable_prefixes: list[str]) -> int:
    if "all" in trainable_prefixes:
        return sum(p.numel() for p in net_g.parameters())
    frozen = 0
    trainable = 0
    for name, param in net_g.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    print(f"Frozen {frozen:,} parameters. Training {trainable:,} parameters ({trainable_prefixes}).")
    return trainable


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_vits_modules(vits_dir: Path):
    sys.path.insert(0, str(vits_dir))
    import commons  # type: ignore
    import losses  # type: ignore
    import mel_processing  # type: ignore
    import models  # type: ignore
    import utils  # type: ignore

    return commons, losses, mel_processing, models, utils


class SingleSpeakerVitsDataset(Dataset):
    def __init__(self, filelist_path: str | Path, config: dict, text_mapper: VocabTextMapper, utils, mel_processing):
        self.rows = []
        with Path(filelist_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                audio_path, text = line.rstrip("\n").split("|", maxsplit=1)
                self.rows.append((audio_path, text))

        self.text_mapper = text_mapper
        self.utils = utils
        self.mel_processing = mel_processing
        self.max_wav_value = float(config["data"]["max_wav_value"])
        self.sampling_rate = int(config["data"]["sampling_rate"])
        self.filter_length = int(config["data"]["filter_length"])
        self.hop_length = int(config["data"]["hop_length"])
        self.win_length = int(config["data"]["win_length"])
        self.add_blank = bool(config["data"]["add_blank"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        audio_path, text = self.rows[index]
        text_ids = torch.LongTensor(self.text_mapper.text_to_ids(text, add_blank=self.add_blank))

        waveform, sample_rate = self.utils.load_wav_to_torch(audio_path)
        if sample_rate != self.sampling_rate:
            raise ValueError(f"{audio_path} has sample rate {sample_rate}, expected {self.sampling_rate}")

        waveform = waveform / self.max_wav_value
        waveform = waveform.unsqueeze(0)

        spec_path = Path(audio_path).with_suffix(".spec.pt")
        if spec_path.exists():
            spectrogram = torch.load(spec_path)
        else:
            spectrogram = self.mel_processing.spectrogram_torch(
                waveform,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False,
            ).squeeze(0)
            torch.save(spectrogram, spec_path)

        return text_ids, spectrogram, waveform


class SingleSpeakerCollate:
    def __call__(self, batch):
        batch = sorted(batch, key=lambda item: item[1].size(1), reverse=True)
        max_text_len = max(item[0].size(0) for item in batch)
        max_spec_len = max(item[1].size(1) for item in batch)
        max_wav_len = max(item[2].size(1) for item in batch)
        spec_channels = batch[0][1].size(0)

        texts = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        text_lengths = torch.zeros(len(batch), dtype=torch.long)
        specs = torch.zeros(len(batch), spec_channels, max_spec_len, dtype=torch.float32)
        spec_lengths = torch.zeros(len(batch), dtype=torch.long)
        waves = torch.zeros(len(batch), 1, max_wav_len, dtype=torch.float32)
        wave_lengths = torch.zeros(len(batch), dtype=torch.long)

        for index, (text, spec, wave) in enumerate(batch):
            texts[index, : text.size(0)] = text
            text_lengths[index] = text.size(0)
            specs[index, :, : spec.size(1)] = spec
            spec_lengths[index] = spec.size(1)
            waves[index, :, : wave.size(1)] = wave
            wave_lengths[index] = wave.size(1)

        return texts, text_lengths, specs, spec_lengths, waves, wave_lengths


def latest_checkpoint(directory: Path, prefix: str) -> Path | None:
    matches = [Path(path) for path in glob.glob(str(directory / f"{prefix}_*.pth"))]
    if not matches:
        return None
    return max(matches, key=lambda path: int("".join(filter(str.isdigit, path.stem)) or 0))


def evaluate(
    config: dict,
    generator,
    eval_loader,
    output_dir: Path,
    global_step: int,
    device,
    mel_processing,
    utils,
    writer: SummaryWriter | None,
):
    generator.eval()
    with torch.no_grad():
        batch = next(iter(eval_loader), None)
        if batch is None:
            return

        texts, text_lengths, specs, spec_lengths, waves, wave_lengths = batch
        texts = texts[:1].to(device)
        text_lengths = text_lengths[:1].to(device)
        specs = specs[:1].to(device)
        spec_lengths = spec_lengths[:1].to(device)
        waves = waves[:1].to(device)
        wave_lengths = wave_lengths[:1].to(device)

        generated, attn, mask, *_ = generator.infer(texts, text_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)
        generated_lengths = mask.sum([1, 2]).long() * int(config["data"]["hop_length"])
        generated_length = int(generated_lengths[0].item())
        reference_length = int(wave_lengths[0].item())

        mel = mel_processing.spec_to_mel_torch(
            specs,
            int(config["data"]["filter_length"]),
            int(config["data"]["n_mel_channels"]),
            int(config["data"]["sampling_rate"]),
            float(config["data"]["mel_fmin"]),
            config["data"]["mel_fmax"],
        )
        mel_generated = mel_processing.mel_spectrogram_torch(
            generated.squeeze(1).float(),
            int(config["data"]["filter_length"]),
            int(config["data"]["n_mel_channels"]),
            int(config["data"]["sampling_rate"]),
            int(config["data"]["hop_length"]),
            int(config["data"]["win_length"]),
            float(config["data"]["mel_fmin"]),
            config["data"]["mel_fmax"],
        )

        eval_dir = output_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        wav_path = eval_dir / f"eval_step_{global_step:06d}.wav"
        write_wav(
            wav_path,
            int(config["data"]["sampling_rate"]),
            generated[0, 0, :generated_length].cpu().float().numpy(),
        )

        if writer is not None:
            utils.summarize(
                writer=writer,
                global_step=global_step,
                images={
                    "eval/mel_generated": utils.plot_spectrogram_to_numpy(mel_generated[0].cpu().numpy()),
                    "eval/mel_reference": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
                    "eval/alignment": utils.plot_alignment_to_numpy(attn[0, 0].cpu().numpy()),
                },
                audios={
                    "eval/audio_generated": generated[0, :, :generated_length],
                    "eval/audio_reference": waves[0, :, :reference_length],
                },
                audio_sampling_rate=int(config["data"]["sampling_rate"]),
            )
    generator.train()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if args.seed is not None:
        config["train"]["seed"] = args.seed
    seed = int(config["train"]["seed"])
    set_seed(seed)

    output_dir = Path(args.output_dir or config["train"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.json")

    commons, losses, mel_processing, models, utils = import_vits_modules(Path(args.vits_dir).resolve())
    text_mapper = VocabTextMapper(config["data"]["vocab_file"])

    train_dataset = SingleSpeakerVitsDataset(config["data"]["training_files"], config, text_mapper, utils, mel_processing)
    val_dataset = SingleSpeakerVitsDataset(config["data"]["validation_files"], config, text_mapper, utils, mel_processing)
    collate = SingleSpeakerCollate()

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        drop_last=len(train_dataset) > int(config["train"]["batch_size"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )

    if not len(train_dataset):
        raise ValueError("Training dataset is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    net_g = models.SynthesizerTrn(
        len(text_mapper.symbols),
        int(config["data"]["filter_length"]) // 2 + 1,
        int(config["train"]["segment_size"]) // int(config["data"]["hop_length"]),
        **config["model"],
    ).to(device)
    net_d = models.MultiPeriodDiscriminator(bool(config["model"]["use_spectral_norm"])).to(device)

    pretrained_dir = Path(args.pretrained_dir).resolve()
    resume_g = latest_checkpoint(output_dir, "G")
    resume_d = latest_checkpoint(output_dir, "D")
    global_step = 0

    if resume_g and resume_d:
        utils.load_checkpoint(str(resume_g), net_g, None)
        utils.load_checkpoint(str(resume_d), net_d, None)
        global_step = int("".join(filter(str.isdigit, resume_g.stem)) or 0)
        print(f"Resuming from step {global_step}.")
    else:
        utils.load_checkpoint(str(pretrained_dir / "G_100000.pth"), net_g, None)
        utils.load_checkpoint(str(pretrained_dir / "D_100000.pth"), net_d, None)

    trainable_prefixes = [p.strip() for p in args.trainable_modules.split(",")]
    freeze_generator(net_g, trainable_prefixes)

    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        lr=float(config["train"]["learning_rate"]),
        betas=tuple(config["train"]["betas"]),
        eps=float(config["train"]["eps"]),
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        lr=float(config["train"]["learning_rate"]),
        betas=tuple(config["train"]["betas"]),
        eps=float(config["train"]["eps"]),
    )

    if resume_g and resume_d:
        utils.load_checkpoint(str(resume_g), net_g, optim_g)
        utils.load_checkpoint(str(resume_d), net_d, optim_d)

    steps_per_epoch = max(1, len(train_dataset) // int(config["train"]["batch_size"]))
    start_epoch = global_step // steps_per_epoch
    # ExponentialLR.step() is called once per epoch; last_epoch=-1 means "not yet stepped",
    # so we pass (start_epoch - 1) to restore the correct LR state when resuming.
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g,
        gamma=float(config["train"]["lr_decay"]),
        last_epoch=start_epoch - 1,
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d,
        gamma=float(config["train"]["lr_decay"]),
        last_epoch=start_epoch - 1,
    )

    _fp16 = bool(config["train"]["fp16_run"]) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler(device_type="cuda", enabled=_fp16)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=_fp16)
    writer = SummaryWriter(log_dir=str(output_dir))
    writer_eval = SummaryWriter(log_dir=str(output_dir / "eval_tb"))

    max_steps = int(config["train"]["max_steps"])
    log_interval = int(config["train"]["log_interval"])
    eval_interval = int(config["train"]["eval_interval"])
    save_interval = int(config["train"]["save_interval"])

    epoch = start_epoch
    while global_step < max_steps:
        net_g.train()
        net_d.train()

        for batch in train_loader:
            if global_step >= max_steps:
                break

            texts, text_lengths, specs, spec_lengths, waves, wave_lengths = batch
            texts = texts.to(device, non_blocking=True)
            text_lengths = text_lengths.to(device, non_blocking=True)
            specs = specs.to(device, non_blocking=True)
            spec_lengths = spec_lengths.to(device, non_blocking=True)
            waves = waves.to(device, non_blocking=True)
            wave_lengths = wave_lengths.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                generated, loss_length, attn, ids_slice, x_mask, z_mask, latent_pack = net_g(
                    texts,
                    text_lengths,
                    specs,
                    spec_lengths,
                )
                z, z_p, m_p, logs_p, m_q, logs_q = latent_pack

                mel = mel_processing.spec_to_mel_torch(
                    specs,
                    int(config["data"]["filter_length"]),
                    int(config["data"]["n_mel_channels"]),
                    int(config["data"]["sampling_rate"]),
                    float(config["data"]["mel_fmin"]),
                    config["data"]["mel_fmax"],
                )
                mel_slice = commons.slice_segments(
                    mel,
                    ids_slice,
                    int(config["train"]["segment_size"]) // int(config["data"]["hop_length"]),
                )
                mel_generated = mel_processing.mel_spectrogram_torch(
                    generated.squeeze(1),
                    int(config["data"]["filter_length"]),
                    int(config["data"]["n_mel_channels"]),
                    int(config["data"]["sampling_rate"]),
                    int(config["data"]["hop_length"]),
                    int(config["data"]["win_length"]),
                    float(config["data"]["mel_fmin"]),
                    config["data"]["mel_fmax"],
                )
                wave_slice = commons.slice_segments(
                    waves,
                    ids_slice * int(config["data"]["hop_length"]),
                    int(config["train"]["segment_size"]),
                )

                real_outputs, fake_outputs, _, _ = net_d(wave_slice, generated.detach())
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    loss_disc, losses_disc_real, losses_disc_fake = losses.discriminator_loss(real_outputs, fake_outputs)

            optim_d.zero_grad(set_to_none=True)
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                real_outputs, fake_outputs, fmap_real, fmap_fake = net_d(wave_slice, generated)
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    loss_duration = torch.sum(loss_length.float())
                    loss_mel = F.l1_loss(mel_slice, mel_generated) * float(config["train"]["c_mel"])
                    loss_kl = losses.kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * float(config["train"]["c_kl"])
                    loss_feature = losses.feature_loss(fmap_real, fmap_fake)
                    loss_generator, generator_terms = losses.generator_loss(fake_outputs)
                    loss_total_g = loss_generator + loss_feature + loss_mel + loss_duration + loss_kl

            optim_g.zero_grad(set_to_none=True)
            scaler.scale(loss_total_g).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            if global_step % log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                print(
                    f"step={global_step} "
                    f"loss_d={loss_disc.item():.4f} "
                    f"loss_g={loss_total_g.item():.4f} "
                    f"mel={loss_mel.item():.4f} "
                    f"dur={loss_duration.item():.4f} "
                    f"kl={loss_kl.item():.4f} "
                    f"lr={lr:.6g}"
                )
                writer.add_scalar("train/loss_d", loss_disc.item(), global_step)
                writer.add_scalar("train/loss_g", loss_total_g.item(), global_step)
                writer.add_scalar("train/loss_mel", loss_mel.item(), global_step)
                writer.add_scalar("train/loss_duration", loss_duration.item(), global_step)
                writer.add_scalar("train/loss_kl", loss_kl.item(), global_step)
                writer.add_scalar("train/loss_feature", loss_feature.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/grad_norm_g", float(grad_norm_g), global_step)
                writer.add_scalar("train/grad_norm_d", float(grad_norm_d), global_step)
                for index, value in enumerate(generator_terms):
                    writer.add_scalar(f"train/generator_term_{index}", value.item(), global_step)
                for index, value in enumerate(losses_disc_real):
                    writer.add_scalar(f"train/disc_real_{index}", value.item(), global_step)
                for index, value in enumerate(losses_disc_fake):
                    writer.add_scalar(f"train/disc_fake_{index}", value.item(), global_step)

            if global_step > 0 and global_step % eval_interval == 0 and len(val_dataset):
                evaluate(config, net_g, val_loader, output_dir, global_step, device, mel_processing, utils, writer_eval)

            if global_step > 0 and global_step % save_interval == 0:
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    float(config["train"]["learning_rate"]),
                    epoch,
                    str(output_dir / f"G_{global_step}.pth"),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    float(config["train"]["learning_rate"]),
                    epoch,
                    str(output_dir / f"D_{global_step}.pth"),
                )

            global_step += 1

        scheduler_g.step()
        scheduler_d.step()
        epoch += 1

    utils.save_checkpoint(
        net_g,
        optim_g,
        float(config["train"]["learning_rate"]),
        epoch,
        str(output_dir / f"G_{global_step}.pth"),
    )
    utils.save_checkpoint(
        net_d,
        optim_d,
        float(config["train"]["learning_rate"]),
        epoch,
        str(output_dir / f"D_{global_step}.pth"),
    )
    writer.close()
    writer_eval.close()
    print(f"Training finished at step {global_step}. Checkpoints are in {output_dir}")


if __name__ == "__main__":
    main()
