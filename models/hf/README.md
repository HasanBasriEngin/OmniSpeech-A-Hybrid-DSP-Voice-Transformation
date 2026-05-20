# Hugging Face Voice Assets

This directory is for optional, locally imported model assets from Hugging Face.

Recommended import command:

```bash
python -m backend.tools.import_hf_voice_assets --bundle all
```

Large downloaded files in this directory are ignored by git. The importer writes
`voice_assets.manifest.json` locally so you can see which repo and revision each
asset came from.

## Local FreeVC Import

If you download or clone `OlaWod/FreeVC` manually, copy the local assets into the
layout OmniSpeech expects:

```powershell
.\.venv\Scripts\python -m backend.tools.import_freevc_assets `
  --source C:\path\to\FreeVC `
  --target models\hf\freevc-24 `
  --variant freevc-24
```

For the original non-24 kHz checkpoint names, use:

```powershell
.\.venv\Scripts\python -m backend.tools.import_freevc_assets `
  --source C:\path\to\FreeVC `
  --target models\hf\freevc-24 `
  --variant freevc `
  --checkpoint C:\path\to\freevc.pth
```

FreeVC is a one-shot conversion engine. The target voice comes from the local
reference audio selected at conversion time, so use only recordings you have
rights or consent to transform.
