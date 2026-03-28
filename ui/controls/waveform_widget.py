from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget


class WaveformWidget(QWidget):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self._audio = np.array([], dtype=np.float32)
        self.setMinimumHeight(140)

    def set_audio(self, audio: np.ndarray) -> None:
        self._audio = np.asarray(audio, dtype=np.float32)
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, QColor("#1a1f31"))
        painter.setPen(QColor("#7080a2"))
        painter.drawText(10, 20, self.title)
        painter.setPen(QPen(QColor("#293552"), 1))
        center_y = rect.height() // 2
        painter.drawLine(0, center_y, rect.width(), center_y)
        if self._audio.size < 2:
            demo = np.linspace(0, 1, max(64, rect.width()), dtype=np.float32)
            self._audio = (0.15 * np.sin(2 * np.pi * demo * 5) + 0.07 * np.sin(2 * np.pi * demo * 14)).astype(
                np.float32
            )
        if self._audio.size < 2:
            return

        data = self._audio
        step = max(1, len(data) // max(1, rect.width()))
        sampled = data[::step]
        max_abs = float(np.max(np.abs(sampled))) + 1e-8
        sampled = sampled / max_abs

        path = QPainterPath()
        width = rect.width()
        for i, val in enumerate(sampled[:width]):
            x = float(i)
            y = center_y - float(val) * (rect.height() * 0.35)
            p = QPointF(x, y)
            if i == 0:
                path.moveTo(p)
            else:
                path.lineTo(p)

        fill_path = QPainterPath(path)
        fill_path.lineTo(rect.width(), center_y)
        fill_path.lineTo(0, center_y)
        fill_path.closeSubpath()
        fill_grad = QLinearGradient(0, 0, rect.width(), 0)
        fill_grad.setColorAt(0.0, QColor(108, 99, 255, 36))
        fill_grad.setColorAt(1.0, QColor(34, 211, 176, 36))
        painter.fillPath(fill_path, fill_grad)

        glow_pen = QPen(QColor(130, 130, 255, 80), 3.2)
        painter.setPen(glow_pen)
        painter.drawPath(path)

        stroke_grad = QLinearGradient(0, 0, rect.width(), 0)
        stroke_grad.setColorAt(0.0, QColor("#6c63ff"))
        stroke_grad.setColorAt(1.0, QColor("#22d3b0"))
        painter.setPen(QPen(stroke_grad, 1.6))
        painter.drawPath(path)
