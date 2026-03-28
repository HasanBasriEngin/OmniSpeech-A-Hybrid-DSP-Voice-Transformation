using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace OmniSpeech_WPF.Converters;

public sealed class PendingToBrushConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        var pending = value is bool b && b;
        return new SolidColorBrush((Color)ColorConverter.ConvertFromString(pending ? "#F59E0B" : "#22D3B0"));
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        throw new NotSupportedException();
    }
}
