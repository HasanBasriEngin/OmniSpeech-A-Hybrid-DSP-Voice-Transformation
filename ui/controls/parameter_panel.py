from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


@dataclass
class ParameterSnapshot:
    mode: str
    option: str
    intensity: float


class ParameterPanel(QWidget):
    def __init__(self, parent=None, show_mode_selector: bool = True) -> None:
        super().__init__(parent)
        self.show_mode_selector = show_mode_selector
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["emotion", "gender_age", "speaker", "singing"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.stack = QStackedWidget()
        self.emotion_combo = QComboBox()
        self.emotion_combo.addItems(["sad", "angry", "excited", "whispered", "calm"])

        self.gender_age_combo = QComboBox()
        self.gender_age_combo.addItems(
            [
                "male_to_female",
                "female_to_male",
                "adult_to_child",
                "adult_to_elderly",
                "child_to_adult",
            ]
        )

        self.speaker_combo = QComboBox()
        self.speaker_combo.addItems(["target_embedding", "clone_from_reference"])

        self.singing_combo = QComboBox()
        self.singing_combo.addItems(["pitch_contour", "midi"])

        self._intensity_label = QLabel("1.00")
        self.intensity_slider = QSlider()
        self.intensity_slider.setOrientation(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(50, 200)
        self.intensity_slider.setValue(100)
        self.intensity_slider.valueChanged.connect(
            lambda v: self._intensity_label.setText(f"{v / 100.0:.2f}")
        )

        self._build_stack()
        self._build_layout()

    def _build_stack(self) -> None:
        for label, widget in [
            ("Emotion Profile", self.emotion_combo),
            ("Gender/Age Map", self.gender_age_combo),
            ("Speaker Mode", self.speaker_combo),
            ("Singing Input", self.singing_combo),
        ]:
            page = QWidget()
            form = QFormLayout(page)
            form.addRow(label, widget)
            self.stack.addWidget(page)

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        mode_group = QGroupBox("Parameters")
        form = QFormLayout(mode_group)
        if self.show_mode_selector:
            form.addRow("Mode", self.mode_combo)
        form.addRow("Parameters", self.stack)
        form.addRow("Intensity", self.intensity_slider)
        form.addRow("Scale", self._intensity_label)
        root.addWidget(mode_group)

    def _on_mode_changed(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)

    def snapshot(self) -> ParameterSnapshot:
        mode = self.mode_combo.currentText()
        option_map = {
            "emotion": self.emotion_combo.currentText(),
            "gender_age": self.gender_age_combo.currentText(),
            "speaker": self.speaker_combo.currentText(),
            "singing": self.singing_combo.currentText(),
        }
        return ParameterSnapshot(
            mode=mode,
            option=option_map[mode],
            intensity=self.intensity_slider.value() / 100.0,
        )

    def set_mode(self, mode: str) -> None:
        idx_map = {"emotion": 0, "gender_age": 1, "speaker": 2, "singing": 3}
        idx = idx_map.get(mode, 0)
        self.mode_combo.setCurrentIndex(idx)
