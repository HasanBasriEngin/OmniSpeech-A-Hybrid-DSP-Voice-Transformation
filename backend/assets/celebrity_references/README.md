# Celebrity Voice Reference Files

Place celebrity reference WAV files in this directory. These are used by the
OpenVoice tone-color converter to clone voice characteristics.

## Expected Files

| Filename               | Celebrity         |
|------------------------|-------------------|
| `michael_jackson.wav`  | Michael Jackson   |
| `morgan_freeman.wav`   | Morgan Freeman    |
| `adele.wav`            | Adele             |
| `james_earl_jones.wav` | James Earl Jones  |
| `taylor_swift.wav`     | Taylor Swift      |

## Requirements

- **Format:** WAV (PCM float32 or int16)
- **Sample rate:** 22050 Hz recommended (will be resampled if different)
- **Duration:** 5–30 seconds of clear speech (no background music or noise)
- **Content:** A clean spoken sentence that captures the speaker's unique vocal
  characteristics (tone, timbre, pitch range)

## Notes

- These files are **not** included in the repository for copyright reasons.
- If a reference file is missing, the system will fall back to DSP-based voice
  conversion methods automatically.
- You can use any short, clean audio clip of the target speaker. Longer clips
  (10–20 s) generally produce better results.
