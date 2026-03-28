using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace OmniSpeech_WPF.Controls;

public partial class PitchContourView : UserControl
{
    public PitchContourView()
    {
        InitializeComponent();
        Loaded += (_, _) => DrawPitch();
        SizeChanged += (_, _) => DrawPitch();
    }

    private void DrawPitch()
    {
        if (ActualWidth <= 1 || ActualHeight <= 1)
        {
            return;
        }

        PitchCanvas.Children.Clear();
        var width = Math.Max(50, (int)ActualWidth);
        var height = Math.Max(30, (int)ActualHeight);

        for (var i = 1; i < 4; i++)
        {
            var y = (height / 4.0) * i;
            PitchCanvas.Children.Add(new Line
            {
                X1 = 0,
                X2 = width,
                Y1 = y,
                Y2 = y,
                Stroke = new SolidColorBrush(Color.FromArgb(45, 255, 255, 255)),
                StrokeThickness = 0.6
            });
        }

        PitchCanvas.Children.Add(BuildContour(width, height, false));
        PitchCanvas.Children.Add(BuildContour(width, height, true));
    }

    private Polyline BuildContour(int width, int height, bool processed)
    {
        var points = new PointCollection();
        for (var x = 0; x < width; x++)
        {
            var t = x / (double)width;
            var shift = processed ? 0.5 : 0.0;
            var y = (processed ? 50 : 55) - (processed ? 42 : 38) *
                (Math.Sin(t * 9 + shift) * 0.4 + Math.Sin(t * 3.5 + shift * 0.6) * 0.35 + Math.Sin(t * 17 + shift * 0.2) * 0.1) *
                Math.Sin(t * Math.PI);
            y = Math.Clamp(y, 4, height - 4);
            points.Add(new Point(x, y));
        }

        return new Polyline
        {
            StrokeThickness = processed ? 2.0 : 1.4,
            Stroke = processed
                ? new LinearGradientBrush(
                    (Color)ColorConverter.ConvertFromString("#6C63FF"),
                    (Color)ColorConverter.ConvertFromString("#22D3B0"),
                    new Point(0, 0),
                    new Point(1, 0))
                : new SolidColorBrush(Color.FromArgb(120, 108, 99, 255)),
            Points = points
        };
    }
}
