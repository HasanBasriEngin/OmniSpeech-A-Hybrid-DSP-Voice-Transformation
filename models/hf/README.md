# Hugging Face Voice Assets

This directory is for optional, locally imported model assets from Hugging Face.

Recommended import command:

```bash
python -m backend.tools.import_hf_voice_assets --bundle all
```

Large downloaded files in this directory are ignored by git. The importer writes
`voice_assets.manifest.json` locally so you can see which repo and revision each
asset came from.
