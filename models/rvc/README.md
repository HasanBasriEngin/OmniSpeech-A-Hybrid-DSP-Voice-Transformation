# Local RVC Models

Put only licensed or consent-based local RVC models here.

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

`registry.json` maps OmniSpeech modes to local model IDs and carries local-only
consent metadata:

```json
{
  "gender_age": {
    "male_to_female": {
      "model_id": "female_local",
      "pitch": 0,
      "index_rate": 0.5,
      "consent_required": true,
      "consent_owner": "authorized_local_voice",
      "license": "private-consent",
      "allow_any_source": false
    }
  },
  "celebrity": {
    "michael_jackson": {
      "model_id": "licensed_profile_local",
      "pitch": 0,
      "index_rate": 0.5,
      "consent_required": true,
      "consent_owner": "licensed_profile_owner",
      "license": "private-consent",
      "allow_any_source": false
    }
  }
}
```

Field notes:

- `consent_required`: defaults to `true` when omitted.
- `consent_owner`: free-text local note for who granted access to the profile.
- `license`: local usage note such as `private-consent`.
- `allow_any_source`: keep `false` unless your process explicitly permits broader input usage.

Import a trained model with the helper script:

```powershell
.\.venv\Scripts\python backend\tools\import_rvc_model.py `
  --model-id my_voice `
  --pth C:\models\my_voice.pth `
  --index C:\models\my_voice.index `
  --category gender_age `
  --key male_to_female `
  --pitch 0 `
  --index-rate 0.5 `
  --consent-required true `
  --consent-owner authorized_local_voice `
  --license private-consent `
  --allow-any-source false
```

The import script only copies local artifacts into `models/rvc/<model_id>/` and
updates `registry.json`. It does not train models and it does not add large
model artifacts to git.

Large model artifacts are intentionally ignored by git.
