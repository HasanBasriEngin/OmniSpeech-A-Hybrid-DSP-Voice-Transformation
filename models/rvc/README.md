# Local RVC Models

Put only licensed or consent-based local RVC models here.

The Hugging Face importer downloads RVC core assets to `models/hf/rvc-core-v2`.
Those files help with RVC tooling, but they are not a target voice. A usable RVC
conversion still needs a local target-speaker `.pth` model and optional `.index`
file that you have rights or consent to use.

Expected layout:

```text
models/rvc/
  registry.json
  female_local/
    female_local.pth
    female_local.index
  licensed_profile_local/
    licensed_profile_local.pth
    licensed_profile_local.index
```

`registry.json` maps OmniSpeech modes to local model IDs:

```json
{
  "gender_age": {
    "male_to_female": {
      "model_id": "female_local",
      "pitch": 0,
      "index_rate": 0.5
    }
  },
  "celebrity": {
    "michael_jackson": {
      "model_id": "licensed_profile_local",
      "pitch": 0,
      "index_rate": 0.5
    }
  }
}
```

Large model artifacts are intentionally ignored by git.
