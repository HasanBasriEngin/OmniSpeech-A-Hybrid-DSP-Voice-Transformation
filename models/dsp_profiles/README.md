# Local DSP AutoTune Profiles

DSP AutoTune stores small numeric post-processing settings here. It does not
store audio. Each conversion updates the profile for its module, even when the
main engine is FreeVC or RVC.

When the main engine is FreeVC or RVC, OmniSpeech applies a conservative neural
post-filter at runtime: no spectral noise reduction, no Pedalboard compressor
chain, no positive make-up gain, and lighter de-essing. The profile still learns
from the run, but the DSP layer avoids overwriting the model output.

Tracked example:

```text
models/dsp_profiles/registry.example.json
```

Local runtime file:

```text
models/dsp_profiles/registry.json
```

The runtime registry is ignored by git because it is machine- and usage-specific.
