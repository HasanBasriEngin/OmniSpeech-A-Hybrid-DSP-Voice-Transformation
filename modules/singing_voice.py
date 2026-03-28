from __future__ import annotations

import numpy as np

from utils.audio_utils import match_length, pitch_shift_simple, time_stretch_simple


def align_to_melody(audio: np.ndarray, sr: int, target_f0: np.ndarray) -> np.ndarray:
    _ = sr
    target_f0 = np.asarray(target_f0, dtype=np.float32)
    voiced = target_f0[target_f0 > 0]
    if voiced.size == 0:
        return np.asarray(audio, dtype=np.float32)
    avg_f0 = float(np.mean(voiced))
    reference = 180.0
    semitones = 12 * np.log2(max(avg_f0, 1.0) / reference)
    return pitch_shift_simple(audio, semitones)


def align_rhythm(audio: np.ndarray, sr: int, beat_times: list[float]) -> np.ndarray:
    if not beat_times:
        return np.asarray(audio, dtype=np.float32)
    target_duration = max(beat_times[-1], 0.2)
    current_duration = len(audio) / float(sr)
    rate = current_duration / target_duration
    return time_stretch_simple(audio, rate=rate)


def apply_singing_timbre(audio: np.ndarray, sr: int) -> np.ndarray:
    _ = sr
    x = np.asarray(audio, dtype=np.float32)
    spec = np.fft.rfft(x)
    freqs = np.linspace(0.0, 1.0, spec.size, dtype=np.float32)
    formant_boost = 1.0 + 0.35 * np.exp(-((freqs - 0.22) ** 2) / (2 * 0.03**2))
    out = np.fft.irfft(spec * formant_boost, n=len(x))
    return np.tanh(out).astype(np.float32)


def load_midi_melody(midi_path: str) -> tuple[np.ndarray, list[float]]:
    try:
        import pretty_midi
    except Exception as exc:
        raise RuntimeError("pretty_midi is required for MIDI melody input.") from exc

    midi = pretty_midi.PrettyMIDI(midi_path)
    f0_values = []
    beat_times = list(midi.get_beats())
    for inst in midi.instruments:
        for note in inst.notes:
            f0_values.append(pretty_midi.note_number_to_hz(note.pitch))
    if not f0_values:
        f0_values = [180.0]
    return np.asarray(f0_values, dtype=np.float32), beat_times


def speech_to_singing(audio: np.ndarray, sr: int, melody_input, input_type: str = "midi") -> np.ndarray:
    if input_type == "midi":
        target_f0, beat_times = load_midi_melody(str(melody_input))
    elif input_type == "pitch_contour":
        target_f0 = np.asarray(melody_input, dtype=np.float32)
        beat_times = [0.0, len(audio) / sr]
    else:
        raise ValueError("input_type must be 'midi' or 'pitch_contour'.")

    out = align_to_melody(audio, sr, target_f0)
    out = align_rhythm(out, sr, beat_times)
    out = apply_singing_timbre(out, sr)
    return match_length(out, len(audio))

