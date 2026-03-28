from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget


class PitchGraph(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._f0 = np.array([], dtype=np.float32)
        self.setMinimumHeight(120)

    def set_f0(self, f0: np.ndarray) -> None:
        self._f0 = np.asarray(f0, dtype=np.float32)
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, QColor("#151a2a"))
        painter.setPen(QColor("#7d88a6"))
        painter.drawText(10, 18, "Pitch Contour (F0)")

        painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
        for i in range(1, 4):
            y = int(rect.height() * i / 4)
            painter.drawLine(0, y, rect.width(), y)

        if self._f0.size < 2:
            x = np.linspace(0, 1, max(64, rect.width()), dtype=np.float32)
            self._f0 = (170 + 32 * np.sin(x * 9) + 18 * np.sin(x * 3.3)).astype(np.float32)
        f0 = self._f0.copy()
        f0[f0 <= 0] = np.nan
        valid = np.where(~np.isnan(f0))[0]
        if valid.size < 2:
            return
        min_v = float(np.nanmin(f0))
        max_v = float(np.nanmax(f0))
        rng = max(max_v - min_v, 1e-6)

        path = QPainterPath()
        for i, idx in enumerate(valid):
            x = (idx / max(1, len(f0) - 1)) * (rect.width() - 1)
            y = rect.height() - 8 - ((f0[idx] - min_v) / rng) * (rect.height() - 24)
            point = QPointF(float(x), float(y))
            if i == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)

        glow_pen = QPen(QColor(120, 120, 255, 80), 3.0)
        painter.setPen(glow_pen)
        painter.drawPath(path)

        grad = QLinearGradient(0, 0, rect.width(), 0)
        grad.setColorAt(0.0, QColor("#6c63ff"))
        grad.setColorAt(1.0, QColor("#22d3b0"))
        painter.setPen(QPen(grad, 1.8))
        painter.drawPath(path)
