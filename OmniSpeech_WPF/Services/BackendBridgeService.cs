using System;
using System.IO;
using System.Threading.Tasks;

namespace OmniSpeech_WPF.Services;

public sealed class BackendBridgeService
{
    private readonly Random _random = new();

    public Task<(string fileName, string metadata)> LoadAudioAsync(string filePath)
    {
        var fi = new FileInfo(filePath);
        var fileName = fi.Name;
        var metadata = $"size {Math.Max(1, fi.Length / 1024)}KB · mono";
        return Task.FromResult((fileName, metadata));
    }

    public async Task<AudioProcessingResult> ProcessAsync(
        string module,
        double pitchRatio,
        double speechRate,
        double energyEnvelope)
    {
        _ = module;
        _ = pitchRatio;
        _ = speechRate;
        _ = energyEnvelope;

        // Baseline bridge: simulates pipeline latency while UI is connected end-to-end.
        await Task.Delay(700);
        return new AudioProcessingResult
        {
            Latency = $"{_random.Next(240, 420)} ms",
            ProcessingTime = $"{_random.NextDouble() * 1.4 + 0.8:0.0} s",
            Fidelity = $"{_random.NextDouble() * 1.2 + 3.3:0.0} /5",
            Intelligibility = $"{_random.NextDouble() * 1.0 + 3.2:0.0} /5",
            ProcessedWaveSeed = _random.NextDouble() * 10
        };
    }
}
