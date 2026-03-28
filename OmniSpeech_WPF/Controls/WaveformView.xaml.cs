using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace OmniSpeech_WPF.Controls;

public partial class WaveformView : UserControl
{
    public static readonly DependencyProperty SeedProperty =
        DependencyProperty.Register(nameof(Seed), typeof(double), typeof(WaveformView),
            new PropertyMetadata(1.0, OnWavePropertyChanged));

    public static readonly DependencyProperty AmplitudeProperty =
        DependencyProperty.Register(nameof(Amplitude), typeof(double), typeof(WaveformView),
            new PropertyMetadata(22.0, OnWavePropertyChanged));

    public static readonly DependencyProperty LabelTextProperty =
        DependencyProperty.Register(nameof(LabelText), typeof(string), typeof(WaveformView),
            new PropertyMetadata("wave"));

    public WaveformView()
    {
        InitializeComponent();
        Loaded += (_, _) => DrawWave();
        SizeChanged += (_, _) => DrawWave();
    }

    public double Seed
    {
        get => (double)GetValue(SeedProperty);
        set => SetValue(SeedProperty, value);
    }

    public double Amplitude
    {
        get => (double)GetValue(AmplitudeProperty);
        set => SetValue(AmplitudeProperty, value);
    }

    public string LabelText
    {
        get => (string)GetValue(LabelTextProperty);
        set => SetValue(LabelTextProperty, value);
    }

    private static void OnWavePropertyChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        ((WaveformView)d).DrawWave();
    }

    private void DrawWave()
    {
        if (ActualWidth <= 1 || ActualHeight <= 1)
        {
            return;
        }

        WaveCanvas.Children.Clear();
        var points = new PointCollection();
        var mid = ActualHeight / 2.0;
        var width = Math.Max(20, (int)ActualWidth);

        for (var x = 0; x < width; x++)
        {
            var t = x / (double)width;
            var y = mid + Amplitude * (
                Math.Sin(t * 28 + Seed) * 0.5 +
                Math.Sin(t * 71 + Seed * 1.3) * 0.25 +
                Math.Sin(t * 130 + Seed * 0.7) * 0.12 +
                Math.Sin(t * 9 + Seed * 2) * 0.13
            ) * Math.Sin(t * Math.PI);
            points.Add(new Point(x, y));
        }

        var polyline = new Polyline
        {
            StrokeThickness = 1.6,
            Stroke = new LinearGradientBrush(
                (Color)ColorConverter.ConvertFromString("#6C63FF"),
                (Color)ColorConverter.ConvertFromString("#22D3B0"),
                new Point(0, 0),
                new Point(1, 0)),
            Points = points
        };
        WaveCanvas.Children.Add(polyline);
    }
}
