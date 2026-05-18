# Local FreeVC Reference Profiles

This directory maps OmniSpeech modes to local, consent-based reference audio for
FreeVC one-shot conversion.

FreeVC does not use a trained target-speaker `.pth/.index` pair like RVC. The
target voice comes from a short local reference recording at conversion time.
For Gender/Age modes, the FreeVC result is used directly by default. The older
extra DSP Gender/Age refinement can be enabled with
`OMNISPEECH_FREEVC_GENDER_AGE_REFINE=1`, but it may introduce metallic or
crackly artifacts on neural model output.

Expected local layout:

```text
models/freevc_profiles/
  registry.json
  references/
    female_local_reference.wav
    older_local_reference.wav
```

Example `registry.json`:

```json
{
  "gender_age": {
    "male_to_female": {
      "profile_id": "female_local_reference",
      "reference_path": "references/female_local_reference.wav",
      "consent_required": true,
      "consent_owner": "authorized_local_voice",
      "license": "private-consent",
      "allow_any_source": false
    }
  }
}
```

Only use local recordings that you have rights or consent to transform. Audio
files in this directory are ignored by git.
