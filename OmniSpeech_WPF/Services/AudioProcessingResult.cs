namespace OmniSpeech_WPF.Services;

public sealed class AudioProcessingResult
{
    public string Latency { get; set; } = "0 ms";
    public string ProcessingTime { get; set; } = "0 s";
    public string Fidelity { get; set; } = "0 /5";
    public string Intelligibility { get; set; } = "0 /5";
    public double ProcessedWaveSeed { get; set; } = 2.4;
}
