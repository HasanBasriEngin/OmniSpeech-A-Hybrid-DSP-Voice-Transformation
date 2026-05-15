from __future__ import annotations

import argparse
import importlib
import logging
import os
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FreeVC 24 kHz inference from imported HF Space assets.")
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--wavlm-model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def _load_external_module(module_name: str):
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def _resolve_path_arg(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_model_arg(raw_model: str, base_dir: Path) -> str:
    path = Path(raw_model).expanduser()
    if path.exists():
        return str(path.resolve())

    candidate = path if path.is_absolute() else base_dir / path
    if candidate.exists():
        return str(candidate.resolve())

    return raw_model


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)

    args = build_parser().parse_args()
    base_dir = Path.cwd()
    assets_dir = _resolve_path_arg(args.assets_dir, base_dir)
    input_path = _resolve_path_arg(args.input, base_dir)
    reference_path = _resolve_path_arg(args.reference, base_dir)
    output_path = _resolve_path_arg(args.output, base_dir)
    wavlm_model = _resolve_model_arg(args.wavlm_model, base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(assets_dir))
    previous_cwd = Path.cwd()
    os.chdir(assets_dir)
    try:
        import librosa
        from scipy.io.wavfile import write
        import torch
        from transformers import WavLMModel

        utils = _load_external_module("utils")
        models = _load_external_module("models")
        speaker_encoder = importlib.import_module("speaker_encoder.voice_encoder")

        requested_device = args.device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        device = torch.device(requested_device)

        hps = utils.get_hparams_from_file("configs/freevc-24.json")
        net_g = models.SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
        ).to(device)
        net_g.eval()
        utils.load_checkpoint("checkpoints/freevc-24.pth", net_g, None)

        speaker_model = speaker_encoder.SpeakerEncoder("speaker_encoder/ckpt/pretrained_bak_5805000.pt")
        wavlm = WavLMModel.from_pretrained(wavlm_model).to(device)
        wavlm.eval()

        with torch.no_grad():
            wav_ref, _ = librosa.load(str(reference_path), sr=hps.data.sampling_rate)
            wav_ref, _ = librosa.effects.trim(wav_ref, top_db=20)
            g_ref = speaker_model.embed_utterance(wav_ref)
            g_ref = torch.from_numpy(g_ref).unsqueeze(0).to(device)

            wav_src, _ = librosa.load(str(input_path), sr=hps.data.sampling_rate)
            wav_src_tensor = torch.from_numpy(wav_src).unsqueeze(0).to(device)
            content = wavlm(wav_src_tensor).last_hidden_state.transpose(1, 2).to(device)
            audio = net_g.infer(content, g=g_ref)
            audio_np = audio[0][0].data.cpu().float().numpy()

        write(str(output_path), 24000, audio_np)
    finally:
        os.chdir(previous_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
