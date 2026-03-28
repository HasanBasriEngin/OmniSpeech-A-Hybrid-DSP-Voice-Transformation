namespace OmniSpeech_WPF.Models;

public sealed class SessionLogEntry
{
    public string Time { get; set; } = "";
    public string Text { get; set; } = "";
    public bool Pending { get; set; }
}
