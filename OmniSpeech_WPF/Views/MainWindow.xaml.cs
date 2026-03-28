using Microsoft.Win32;
using OmniSpeech_WPF.Models;
using OmniSpeech_WPF.Services;
using System;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;

namespace OmniSpeech_WPF.Views;

public partial class MainWindow : Window
{
    private readonly AppState _state = new();
    private readonly BackendBridgeService _backendBridge = new();
    private readonly DispatcherTimer _playTimer = new();
    private bool _playing;
    private double _playbackPercent = 34;

    public MainWindow()
    {
        InitializeComponent();
        DataContext = _state;

        _state.SessionLogs.Add(new SessionLogEntry { Time = "14:32", Text = "sample_speech.wav loaded (2.8s)", Pending = false });
        _state.SessionLogs.Add(new SessionLogEntry { Time = "14:33", Text = "F0 tracking complete - 187Hz mean", Pending = true });

        _playTimer.Interval = TimeSpan.FromMilliseconds(35);
        _playTimer.Tick += (_, _) => TickPlayback();

        UpdateModuleSelectionVisuals();
    }

    private async void DropZone_Click(object sender, System.Windows.Input.MouseButtonEventArgs e)
    {
        _ = sender;
        _ = e;
        var dialog = new OpenFileDialog
        {
            Filter = "Audio Files (*.wav;*.mp3;*.flac)|*.wav;*.mp3;*.flac|All Files (*.*)|*.*"
        };
        if (dialog.ShowDialog() != true)
        {
            return;
        }

        var (fileName, meta) = await _backendBridge.LoadAudioAsync(dialog.FileName);
        _state.LoadedFileName = $"OK {fileName}";
        _state.LoadedFileMeta = meta;
        AppendLog($"{fileName} loaded");
    }

    private void PlayPauseButton_Click(object sender, RoutedEventArgs e)
    {
        _ = sender;
        _ = e;
        _playing = !_playing;
        PlayPauseButton.Content = _playing ? "❚❚" : "▶";
        if (_playing)
        {
            _playTimer.Start();
        }
        else
        {
            _playTimer.Stop();
        }
    }

    private void TickPlayback()
    {
        _playbackPercent += 0.15;
        if (_playbackPercent > 100)
        {
            _playbackPercent = 0;
        }
        PlaybackProgress.Value = _playbackPercent;
        var seconds = (int)Math.Floor((_playbackPercent / 100.0) * 2.8);
        TimeText.Text = $"0:0{seconds} / 0:02";
    }

    private async void ConvertButton_Click(object sender, RoutedEventArgs e)
    {
        _ = sender;
        _ = e;
        ConvertButton.IsEnabled = false;
        ConvertButton.Content = "Processing...";
        AppendLog($"{_state.SelectedModule} module initialized", true);

        var result = await _backendBridge.ProcessAsync(_state.SelectedModule, _state.PitchRatio, _state.SpeechRate, _state.EnergyEnvelope);
        _state.Latency = result.Latency;
        _state.ProcTime = result.ProcessingTime;
        _state.Fidelity = result.Fidelity;
        _state.Intelligibility = result.Intelligibility;
        ProcessedWaveView.Seed = result.ProcessedWaveSeed;
        AppendLog("Output generated - ready for export");

        ConvertButton.IsEnabled = true;
        ConvertButton.Content = "▶ Convert Audio";
    }

    private void EmotionModuleButton_Click(object sender, RoutedEventArgs e) => SelectModule("emotion");
    private void GenderModuleButton_Click(object sender, RoutedEventArgs e) => SelectModule("gender");
    private void SpeakerModuleButton_Click(object sender, RoutedEventArgs e) => SelectModule("speaker");
    private void SingingModuleButton_Click(object sender, RoutedEventArgs e) => SelectModule("singing");

    private void SelectModule(string module)
    {
        _state.SelectedModule = module;
        UpdateModuleSelectionVisuals();
    }

    private void UpdateModuleSelectionVisuals()
    {
        ResetModuleButton(EmotionModuleButton);
        ResetModuleButton(GenderModuleButton);
        ResetModuleButton(SpeakerModuleButton);
        ResetModuleButton(SingingModuleButton);

        var active = _state.SelectedModule switch
        {
            "emotion" => EmotionModuleButton,
            "gender" => GenderModuleButton,
            "speaker" => SpeakerModuleButton,
            "singing" => SingingModuleButton,
            _ => EmotionModuleButton
        };
        active.BorderBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#6C63FF"));
        active.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#2A2D4F"));
        active.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#A78BFA"));
    }

    private static void ResetModuleButton(Button btn)
    {
        btn.BorderBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#2C3346"));
        btn.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1C2030"));
        btn.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#8B92A8"));
    }

    private void EmotionChip_Click(object sender, RoutedEventArgs e)
    {
        if (sender is Button b && b.Tag is string emotion)
        {
            _state.TargetEmotion = emotion;
            AppendLog($"Target emotion selected: {emotion}");
        }
    }

    private void AppendLog(string text, bool pending = false)
    {
        _state.SessionLogs.Add(new SessionLogEntry
        {
            Time = DateTime.Now.ToString("HH:mm"),
            Text = text,
            Pending = pending
        });
    }
}
