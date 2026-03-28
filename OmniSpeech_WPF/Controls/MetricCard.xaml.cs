using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace OmniSpeech_WPF.Controls;

public partial class MetricCard : UserControl
{
    public static readonly DependencyProperty LabelProperty =
        DependencyProperty.Register(nameof(Label), typeof(string), typeof(MetricCard), new PropertyMetadata(""));

    public static readonly DependencyProperty ValueProperty =
        DependencyProperty.Register(nameof(Value), typeof(string), typeof(MetricCard), new PropertyMetadata(""));

    public static readonly DependencyProperty ValueBrushProperty =
        DependencyProperty.Register(nameof(ValueBrush), typeof(Brush), typeof(MetricCard),
            new PropertyMetadata(new SolidColorBrush((Color)ColorConverter.ConvertFromString("#E8EAF0"))));

    public MetricCard()
    {
        InitializeComponent();
    }

    public string Label
    {
        get => (string)GetValue(LabelProperty);
        set => SetValue(LabelProperty, value);
    }

    public string Value
    {
        get => (string)GetValue(ValueProperty);
        set => SetValue(ValueProperty, value);
    }

    public Brush ValueBrush
    {
        get => (Brush)GetValue(ValueBrushProperty);
        set => SetValue(ValueBrushProperty, value);
    }
}
