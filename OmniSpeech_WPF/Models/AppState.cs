using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace OmniSpeech_WPF.Models;

public sealed class AppState : INotifyPropertyChanged
{
    private string _loadedFileName = "Drop audio file here";
    private string _loadedFileMeta = "or click to browse";
    private string _selectedModule = "emotion";
    private string _targetEmotion = "angry";
    private double _pitchRatio = 1.5;
    private double _speechRate = 1.15;
    private double _energyEnvelope = 1.4;
    private string _latency = "312 ms";
    private string _procTime = "1.4 s";
    private string _fidelity = "4.2 /5";
    private string _intelligibility = "3.8 /5";

    public string LoadedFileName
    {
        get => _loadedFileName;
        set => SetField(ref _loadedFileName, value);
    }

    public string LoadedFileMeta
    {
        get => _loadedFileMeta;
        set => SetField(ref _loadedFileMeta, value);
    }

    public string SelectedModule
    {
        get => _selectedModule;
        set => SetField(ref _selectedModule, value);
    }

    public string TargetEmotion
    {
        get => _targetEmotion;
        set => SetField(ref _targetEmotion, value);
    }

    public double PitchRatio
    {
        get => _pitchRatio;
        set => SetField(ref _pitchRatio, value);
    }

    public double SpeechRate
    {
        get => _speechRate;
        set => SetField(ref _speechRate, value);
    }

    public double EnergyEnvelope
    {
        get => _energyEnvelope;
        set => SetField(ref _energyEnvelope, value);
    }

    public string Latency
    {
        get => _latency;
        set => SetField(ref _latency, value);
    }

    public string ProcTime
    {
        get => _procTime;
        set => SetField(ref _procTime, value);
    }

    public string Fidelity
    {
        get => _fidelity;
        set => SetField(ref _fidelity, value);
    }

    public string Intelligibility
    {
        get => _intelligibility;
        set => SetField(ref _intelligibility, value);
    }

    public ObservableCollection<SessionLogEntry> SessionLogs { get; } = new();

    public event PropertyChangedEventHandler? PropertyChanged;

    private void SetField<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (Equals(field, value))
        {
            return;
        }

        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}
