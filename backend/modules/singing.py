from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from backend.audio.features import extract_pitch_contour, stretch_to_length


def _target_pitch_hz(midi_path: str | None, pitch_contour: list[float] | None) -> float:
    if midi_path:
        try:
            import pretty_midi
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pretty_midi is required for MIDI based singing conversion.") from exc

        midi = pretty_midi.PrettyMIDI(str(Path(midi_path).expanduser().resolve()))
        notes_hz: list[float] = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                notes_hz.append(float(pretty_midi.note_number_to_hz(note.pitch)))
        if notes_hz:
            return float(np.median(np.asarray(notes_hz, dtype=np.float32)))

    if pitch_contour:
        contour = np.asarray(pitch_contour, dtype=np.float32)
        voiced = contour[contour > 0]
        if voiced.size:
            return float(np.median(voiced))

    return 220.0


def convert_to_singing(
    audio: np.ndarray,
    sample_rate: int,
    midi_path: str | None,
    pitch_contour: list[float] | None,
) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    source_f0 = extract_pitch_contour(x, sample_rate)
    voiced = source_f0[source_f0 > 0]
    source_hz = float(np.median(voiced)) if voiced.size else 180.0
    target_hz = _target_pitch_hz(midi_path, pitch_contour)

    semitone_delta = 12.0 * np.log2(max(target_hz, 1.0) / max(source_hz, 1.0))
    pitched = librosa.effects.pitch_shift(y=x, sr=sample_rate, n_steps=float(semitone_delta))

    stretched = librosa.effects.time_stretch(y=np.asarray(pitched, dtype=np.float32), rate=0.92)

    # Mild spectral brightening to emulate sung phonation.
    spec = np.fft.rfft(stretched)
    freqs = np.linspace(0.0, 1.0, spec.size, dtype=np.float32)
    tilt = 1.0 + 0.35 * np.exp(-((freqs - 0.28) ** 2) / (2 * 0.06**2))
    bright = np.fft.irfft(spec * tilt, n=stretched.size).astype(np.float32)

    out = stretch_to_length(bright, x.size)
    return np.clip(out, -1.0, 1.0).astype(np.float32)
