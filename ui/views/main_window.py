from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.preprocessing import detect_f0
from ui.controls.parameter_panel import ParameterSnapshot
from ui.controls.pitch_graph import PitchGraph
from ui.controls.waveform_widget import WaveformWidget
from ui.converters.format_converter import short_path
from ui.models.app_state import AppState
from ui.services.backend_bridge import BackendBridge
from utils.audio_utils import match_length, pitch_shift_simple, time_stretch_simple


def _card(name: str = "Card") -> QFrame:
    frame = QFrame()
    frame.setObjectName(name)
    return frame


def _slider_row(title: str, value_label: QLabel, slider: QSlider, value_color: str) -> QWidget:
    wrap = QWidget()
    lay = QVBoxLayout(wrap)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)

    head = QHBoxLayout()
    title_label = QLabel(title)
    title_label.setObjectName("ParamName")
    value_label.setObjectName("ParamValue")
    value_label.setStyleSheet(f"color: {value_color};")
    head.addWidget(title_label)
    head.addStretch(1)
    head.addWidget(value_label)
    lay.addLayout(head)
    lay.addWidget(slider)
    return wrap


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.state = AppState()
        self.backend = BackendBridge(self.state)

        self.input_mode = "file"
        self.current_mode = "emotion"
        self.current_emotion = "angry"

        self._is_playing = False
        self._duration_sec = 0.0
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._tick_playback)
        self._play_progress = 0
        self._module_btn_height = 70

        self.setWindowTitle("SpeechWarp")
        self.resize(1560, 930)
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QWidget()
        shell = QVBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        shell.addWidget(self._build_topbar(), 0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        shell.addLayout(body, 1)

        body.addWidget(self._build_sidebar(), 0)
        body.addWidget(self._build_center(), 1)
        body.addWidget(self._build_right_panel(), 0)

        self.setCentralWidget(root)
        self._apply_theme()
        self._set_mode("emotion")
        self._set_emotion("angry")
        self._append_log("UI ready")

    def _build_topbar(self) -> QWidget:
        bar = _card("TopBar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(8)

        logo = QLabel("SpeechWarp\nv0.9 | CENG 384")
        logo.setObjectName("Logo")
        lay.addWidget(logo)
        lay.addStretch(1)
        lay.addWidget(QLabel("* ready"))
        lay.addWidget(QLabel("CPU | 12% load"))
        return bar

    def _build_sidebar(self) -> QWidget:
        self.sidebar_panel = _card("Sidebar")
        self.sidebar_panel.setFixedWidth(230)
        lay = QVBoxLayout(self.sidebar_panel)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(8)

        for cap, entries in [("NAVIGATION", ["Workspace"]), ("SYSTEM", ["Evaluation", "Settings"])]:
            c = QLabel(cap)
            c.setObjectName("SectionCap")
            lay.addWidget(c)
            for entry in entries:
                btn = QPushButton(entry)
                btn.setObjectName("SideBtn")
                btn.clicked.connect(lambda _=False, name=entry: self._append_log(f"{name} opened"))
                lay.addWidget(btn)

        lay.addStretch(1)
        flow = QLabel("3-STEP FLOW\n[x] Load audio\n2 Select module\n3 Convert & export")
        flow.setObjectName("FlowText")
        lay.addWidget(flow)
        return self.sidebar_panel

    def _build_center(self) -> QWidget:
        center = QWidget()
        lay = QVBoxLayout(center)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(14)

        input_card = _card()
        input_lay = QVBoxLayout(input_card)
        input_lay.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(QLabel("// audio_input"))
        header.addStretch(1)
        self.file_mode_btn = QPushButton("FILE")
        self.file_mode_btn.setObjectName("MiniActive")
        self.file_mode_btn.clicked.connect(lambda: self._set_input_mode("file"))
        self.mic_mode_btn = QPushButton("MIC")
        self.mic_mode_btn.setObjectName("MiniBtn")
        self.mic_mode_btn.clicked.connect(lambda: self._set_input_mode("mic"))
        header.addWidget(self.file_mode_btn)
        header.addWidget(self.mic_mode_btn)
        input_lay.addLayout(header)

        drop = _card("DropZone")
        dlay = QVBoxLayout(drop)
        dlay.setContentsMargins(10, 10, 10, 10)
        dlay.setSpacing(4)
        self.path_label = QLabel("Drop audio file here")
        self.path_label.setObjectName("DropTitle")
        dlay.addWidget(self.path_label)
        self.drop_sub = QLabel("or click to browse")
        dlay.addWidget(self.drop_sub)
        dlay.addWidget(QLabel("WAV   MP3   FLAC   16kHz/22kHz"))
        self.load_btn = QPushButton("Load Audio")
        self.load_btn.setObjectName("LoadBtn")
        self.load_btn.clicked.connect(self._on_load)
        dlay.addWidget(self.load_btn)
        input_lay.addWidget(drop)

        self.wave_in = WaveformWidget("original")
        self.wave_out = WaveformWidget("processed")
        input_lay.addWidget(self.wave_in)
        input_lay.addWidget(self.wave_out)

        play_row = QHBoxLayout()
        self.play_btn = QPushButton(">")
        self.play_btn.setObjectName("PlayBtn")
        self.play_btn.clicked.connect(self._toggle_playback)
        play_row.addWidget(self.play_btn)
        self.playback = QProgressBar()
        self.playback.setObjectName("Playback")
        self.playback.setRange(0, 100)
        self.playback.setValue(0)
        play_row.addWidget(self.playback, 1)
        self.time_label = QLabel("0:00 / 0:00")
        play_row.addWidget(self.time_label)
        input_lay.addLayout(play_row)
        lay.addWidget(input_card)

        pitch_card = _card()
        pitch_lay = QVBoxLayout(pitch_card)
        pitch_lay.addWidget(QLabel("// f0_pitch_contour"))
        self.pitch_graph = PitchGraph()
        pitch_lay.addWidget(self.pitch_graph)
        lay.addWidget(pitch_card)

        metrics_wrap = QWidget()
        grid = QGridLayout(metrics_wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)

        self.metric_latency = QLabel("0 ms")
        self.metric_proc = QLabel("0.0 s")
        self.metric_fid = QLabel("0.0 /5")
        self.metric_int = QLabel("0.0 /5")
        self.metric_latency.setStyleSheet("color:#22d3b0;")
        self.metric_proc.setStyleSheet("color:#22d3b0;")
        self.metric_int.setStyleSheet("color:#22d3b0;")

        metric_items = [
            ("LATENCY", self.metric_latency),
            ("PROC TIME", self.metric_proc),
            ("FIDELITY", self.metric_fid),
            ("INTELLIG.", self.metric_int),
        ]
        for i, (cap, value) in enumerate(metric_items):
            card = _card("MetricCard")
            cl = QVBoxLayout(card)
            cl.setSpacing(2)
            cap_label = QLabel(cap)
            cap_label.setObjectName("MetricCap")
            value.setObjectName("MetricVal")
            cl.addWidget(cap_label)
            cl.addWidget(value)
            grid.addWidget(card, i // 2, i % 2)
        lay.addWidget(metrics_wrap)
        return center

    def _build_right_panel(self) -> QWidget:
        self.right_panel = _card("RightPanel")
        self.right_panel.setFixedWidth(340)
        lay = QVBoxLayout(self.right_panel)
        lay.setContentsMargins(14, 16, 14, 16)
        lay.setSpacing(10)

        lay.addWidget(QLabel("// MODULE_SELECT"))
        mod_grid = QGridLayout()
        mod_grid.setSpacing(8)
        self._module_buttons: dict[str, QPushButton] = {
            "emotion": QPushButton("[E]\nEmotion"),
            "gender_age": QPushButton("[G]\nGender/Age"),
            "speaker": QPushButton("[S]\nSpeaker"),
            "singing": QPushButton("[N]\nSinging"),
        }
        for mode, btn in self._module_buttons.items():
            btn.setObjectName("ModuleBtn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, m=mode: self._set_mode(m))
        mod_grid.addWidget(self._module_buttons["emotion"], 0, 0)
        mod_grid.addWidget(self._module_buttons["gender_age"], 0, 1)
        mod_grid.addWidget(self._module_buttons["speaker"], 1, 0)
        mod_grid.addWidget(self._module_buttons["singing"], 1, 1)
        lay.addLayout(mod_grid)

        lay.addWidget(QLabel("// PARAMETERS"))

        self.pitch_label = QLabel("+1.5 st")
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setObjectName("PitchSlider")
        self.pitch_slider.setRange(-60, 60)
        self.pitch_slider.setValue(15)
        self.pitch_slider.valueChanged.connect(self._refresh_param_labels)
        lay.addWidget(_slider_row("Pitch Ratio", self.pitch_label, self.pitch_slider, "#a78bfa"))

        self.rate_label = QLabel("1.15x")
        self.rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.rate_slider.setObjectName("RateSlider")
        self.rate_slider.setRange(60, 150)
        self.rate_slider.setValue(115)
        self.rate_slider.valueChanged.connect(self._refresh_param_labels)
        lay.addWidget(_slider_row("Speech Rate", self.rate_label, self.rate_slider, "#22d3b0"))

        self.energy_label = QLabel("1.40x")
        self.energy_slider = QSlider(Qt.Orientation.Horizontal)
        self.energy_slider.setObjectName("EnergySlider")
        self.energy_slider.setRange(20, 200)
        self.energy_slider.setValue(140)
        self.energy_slider.valueChanged.connect(self._refresh_param_labels)
        lay.addWidget(_slider_row("Energy Envelope", self.energy_label, self.energy_slider, "#fbbf24"))

        lay.addWidget(QLabel("// TARGET_EMOTION"))
        chips = QHBoxLayout()
        chips.setSpacing(6)
        self._emotion_buttons: dict[str, QPushButton] = {}
        for emotion in ["sad", "angry", "excited", "whisper", "calm"]:
            b = QPushButton(emotion)
            b.setObjectName("EmotionChip")
            b.clicked.connect(lambda _=False, e=emotion: self._set_emotion(e))
            self._emotion_buttons[emotion] = b
            chips.addWidget(b)
        lay.addLayout(chips)

        self.convert_btn = QPushButton("> Convert Audio")
        self.convert_btn.setObjectName("ConvertBtn")
        self.convert_btn.clicked.connect(self._on_transform)
        lay.addWidget(self.convert_btn)

        exports = QGridLayout()
        self.export_audio_btn = QPushButton("Export Audio")
        self.export_audio_btn.clicked.connect(self._on_export_audio)
        self.export_embedding_btn = QPushButton("Export Embedding")
        self.export_embedding_btn.clicked.connect(self._on_export_embedding)
        self.save_log_btn = QPushButton("Save Session Log")
        self.save_log_btn.clicked.connect(self._on_save_log)
        exports.addWidget(self.export_audio_btn, 0, 0)
        exports.addWidget(self.export_embedding_btn, 0, 1)
        exports.addWidget(self.save_log_btn, 1, 0, 1, 2)
        lay.addLayout(exports)

        lay.addWidget(QLabel("// SESSION_LOG"))
        self.log_box = QTextEdit()
        self.log_box.setObjectName("LogBox")
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Session log")
        lay.addWidget(self.log_box, 1)

        self._refresh_param_labels()
        return self.right_panel

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background:#0d0f12; color:#e8eaf0; font-size:13px; }
            QFrame#TopBar { background:#141720; border-bottom:1px solid #252d43; }
            QLabel#Logo { font-size:14px; font-weight:700; color:#ffffff; }

            QFrame#Sidebar { background:#111522; border-right:1px solid #212944; }
            QLabel#SectionCap { color:#5a6588; font-size:10px; letter-spacing:0.08em; margin-top:4px; }
            QPushButton#SideBtn {
                text-align:left; padding:8px 10px; border-radius:8px; border:1px solid transparent;
                background:transparent; color:#9ca7c8;
            }
            QPushButton#SideBtn:hover { background:#1b2238; border-color:#313b5a; color:#f2f5ff; }
            QLabel#FlowText { color:#93a2c7; border-top:1px solid #29314a; padding-top:10px; }

            QFrame#Card { background:#141a2a; border:1px solid #26304b; border-radius:12px; }
            QFrame#DropZone { background:#111827; border:1px dashed #4a5679; border-radius:10px; min-height:170px; }
            QLabel#DropTitle { font-size:32px; font-weight:600; color:#f4f6ff; }
            QPushButton#LoadBtn {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #6c63ff, stop:1 #22d3b0);
                border:none; border-radius:10px; padding:10px 16px; color:#fff; font-weight:700;
            }
            QPushButton#MiniBtn, QPushButton#MiniActive {
                background:#1d2437; border:1px solid #38415f; border-radius:7px; padding:4px 11px; color:#94a3c9;
            }
            QPushButton#MiniActive { background:#2b2e5a; border-color:#6c63ff; color:#c3baff; }
            QPushButton#PlayBtn { background:#1b2238; border:1px solid #384663; border-radius:16px; width:32px; height:32px; }
            QProgressBar#Playback { background:#222c47; border:1px solid #2e3957; border-radius:4px; height:8px; color:#8f9dc1; }
            QProgressBar::chunk { background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #6c63ff, stop:1 #22d3b0); border-radius:4px; }

            QFrame#RightPanel { background:#111726; border-left:1px solid #212944; }
            QPushButton#ModuleBtn {
                min-height:70px; background:#1b2238; border:1px solid #313d5a;
                border-radius:12px; color:#96a4c9; font-weight:600;
            }
            QPushButton#ModuleBtn:hover { border-color:#6c63ff; color:#d4ccff; }

            QLabel#ParamName { color:#a5b0ce; font-size:12px; }
            QLabel#ParamValue { font-size:28px; font-weight:700; }
            QSlider::groove:horizontal { background:#2b3552; height:4px; border-radius:2px; }
            QSlider::handle:horizontal {
                background:#a78bfa; width:14px; margin:-5px 0; border-radius:7px; border:1px solid #141a2a;
            }
            QSlider#RateSlider::handle:horizontal { background:#22d3b0; }
            QSlider#EnergySlider::handle:horizontal { background:#fbbf24; }

            QPushButton#EmotionChip {
                background:#121a2d; border:1px solid #2d3651; border-radius:14px; padding:4px 10px; color:#8fa0c8;
            }
            QPushButton#EmotionChip:hover { border-color:#6c63ff; color:#cfc6ff; }

            QPushButton#ConvertBtn {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #6c63ff, stop:1 #8a5df8);
                border:none; border-radius:12px; color:#fff; font-size:22px; font-weight:700; padding:11px;
            }

            QFrame#MetricCard { background:#1b2238; border:1px solid #2f3958; border-radius:12px; }
            QLabel#MetricCap { color:#5d6b92; font-size:10px; letter-spacing:0.11em; }
            QLabel#MetricVal { color:#f2f6ff; font-size:38px; font-weight:700; }
            QTextEdit#LogBox { background:#151d2f; border:1px solid #29334d; border-radius:10px; color:#b8c1db; }
            QPushButton { background:#1d2437; border:1px solid #34415f; border-radius:8px; padding:8px 10px; color:#c8d1e9; }
            QPushButton:hover { border-color:#6c63ff; color:#e0d9ff; }
            """
        )
        self._update_responsive_sizes()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_responsive_sizes()

    def _update_responsive_sizes(self) -> None:
        total_w = max(self.width(), 900)
        total_h = max(self.height(), 640)

        sidebar_w = max(190, min(260, int(total_w * 0.14)))
        right_w = max(300, min(390, int(total_w * 0.22)))
        self.sidebar_panel.setFixedWidth(sidebar_w)
        self.right_panel.setFixedWidth(right_w)

        center_w = max(600, total_w - sidebar_w - right_w - 40)
        drop_px = max(24, min(42, int(center_w * 0.045)))
        self.path_label.setStyleSheet(
            f"font-size:{drop_px}px; font-weight:600; color:#f4f6ff;"
        )

        metric_px = max(26, min(54, int(center_w * 0.05)))
        self.metric_latency.setStyleSheet(f"font-size:{metric_px}px; font-weight:700; color:#22d3b0;")
        self.metric_proc.setStyleSheet(f"font-size:{metric_px}px; font-weight:700; color:#22d3b0;")
        self.metric_fid.setStyleSheet(f"font-size:{metric_px}px; font-weight:700; color:#f2f6ff;")
        self.metric_int.setStyleSheet(f"font-size:{metric_px}px; font-weight:700; color:#22d3b0;")

        self._module_btn_height = max(58, min(84, int(total_h * 0.085)))
        for btn in self._module_buttons.values():
            btn.setMinimumHeight(self._module_btn_height)
            btn.setMaximumHeight(self._module_btn_height)

    def _set_input_mode(self, mode: str) -> None:
        self.input_mode = mode
        if mode == "file":
            self.file_mode_btn.setObjectName("MiniActive")
            self.mic_mode_btn.setObjectName("MiniBtn")
            self.load_btn.setText("Load Audio")
            self.drop_sub.setText("or click to browse")
        else:
            self.file_mode_btn.setObjectName("MiniBtn")
            self.mic_mode_btn.setObjectName("MiniActive")
            self.load_btn.setText("Capture Mic Sample")
            self.drop_sub.setText("capture from microphone (simulated)")
        self._apply_theme()
        self._append_log(f"Input mode: {mode}")

    def _set_mode(self, mode: str) -> None:
        self.current_mode = mode
        for m, btn in self._module_buttons.items():
            if m == mode:
                btn.setStyleSheet(
                    f"min-height:{self._module_btn_height}px; max-height:{self._module_btn_height}px; "
                    "background:#2a2f52; border:1px solid #6c63ff; border-radius:12px; color:#d4ccff; font-weight:700;"
                )
            else:
                btn.setStyleSheet("")
        self._append_log(f"Module selected: {mode}")

    def _set_emotion(self, emotion: str) -> None:
        self.current_emotion = "whispered" if emotion == "whisper" else emotion
        for e, btn in self._emotion_buttons.items():
            if e == emotion:
                btn.setStyleSheet(
                    "background:#2a1c2a; border:1px solid #f87171; border-radius:14px; padding:4px 10px; color:#ff7575;"
                )
            else:
                btn.setStyleSheet("")
        self._append_log(f"Target emotion: {emotion}")

    def _refresh_param_labels(self) -> None:
        self.pitch_label.setText(f"{self.pitch_slider.value()/10:+.1f} st")
        self.rate_label.setText(f"{self.rate_slider.value()/100:.2f}x")
        self.energy_label.setText(f"{self.energy_slider.value()/100:.2f}x")

    def _on_load(self) -> None:
        if self.input_mode == "mic":
            self._capture_mic_sample()
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            "",
            "Audio Files (*.wav *.mp3 *.flac)",
        )
        if not path:
            return
        try:
            self.backend.load_input(path)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self._duration_sec = len(self.state.input_audio) / float(self.state.sample_rate)
        self.path_label.setText(short_path(path))
        self.wave_in.set_audio(self.state.input_audio)
        self.wave_out.set_audio(self.state.input_audio)
        self.time_label.setText(f"0:00 / {self._duration_sec:0.2f}")
        self._append_log(f"[input] {Path(path).name} loaded")

    def _capture_mic_sample(self) -> None:
        sr = self.state.sample_rate
        self._duration_sec = 2.8
        t = np.linspace(0, self._duration_sec, int(sr * self._duration_sec), endpoint=False, dtype=np.float32)
        audio = 0.24 * np.sin(2 * np.pi * 180 * t) + 0.11 * np.sin(2 * np.pi * 340 * t)
        audio *= np.linspace(0.9, 1.1, audio.size, dtype=np.float32)
        self.state.input_audio = audio.astype(np.float32)
        self.path_label.setText("mic_capture.wav (simulated)")
        self.wave_in.set_audio(self.state.input_audio)
        self.wave_out.set_audio(self.state.input_audio)
        self.time_label.setText(f"0:00 / {self._duration_sec:0.2f}")
        self._append_log("[input] mic sample captured")

    def _toggle_playback(self) -> None:
        if self.state.input_audio.size == 0 and self.state.processed_audio.size == 0:
            self._append_log("No audio to play")
            return
        self._is_playing = not self._is_playing
        if self._is_playing:
            self.play_btn.setText("||")
            self._play_timer.start(60)
            self._append_log("Playback started")
        else:
            self.play_btn.setText(">")
            self._play_timer.stop()
            self._append_log("Playback paused")

    def _tick_playback(self) -> None:
        self._play_progress = (self._play_progress + 2) % 101
        self.playback.setValue(self._play_progress)
        current = (self._play_progress / 100.0) * max(self._duration_sec, 0.01)
        self.time_label.setText(f"{current:0.2f} / {max(self._duration_sec, 0.01):0.2f}")
        if self._play_progress >= 100:
            self._is_playing = False
            self.play_btn.setText(">")
            self._play_timer.stop()

    def _build_params(self) -> ParameterSnapshot:
        if self.current_mode == "emotion":
            option = self.current_emotion
        elif self.current_mode == "gender_age":
            option = "male_to_female" if self.pitch_slider.value() >= 0 else "female_to_male"
        elif self.current_mode == "speaker":
            option = "target_embedding"
        else:
            option = "pitch_contour"
        intensity = self.energy_slider.value() / 100.0
        return ParameterSnapshot(mode=self.current_mode, option=option, intensity=intensity)

    def _apply_post_controls(self, audio: np.ndarray) -> np.ndarray:
        pitch_st = self.pitch_slider.value() / 10.0
        rate = self.rate_slider.value() / 100.0
        energy = self.energy_slider.value() / 100.0

        out = np.asarray(audio, dtype=np.float32)
        if abs(pitch_st) > 0.01:
            out = pitch_shift_simple(out, pitch_st)
        if abs(rate - 1.0) > 0.01:
            out = time_stretch_simple(out, rate)
            out = match_length(out, len(self.state.input_audio))
        out = np.clip(out * energy, -1.0, 1.0).astype(np.float32)
        return out

    def _on_transform(self) -> None:
        if self.state.input_audio.size == 0:
            self._show_error("Load audio or capture mic sample first.")
            return

        self.playback.setValue(15)
        QApplication.processEvents()
        try:
            params = self._build_params()
            self.playback.setValue(45)
            QApplication.processEvents()
            out_audio, _, metrics = self.backend.process(params)
            out_audio = self._apply_post_controls(out_audio)
            f0 = detect_f0(out_audio, self.state.sample_rate)
            self.playback.setValue(100)
        except Exception as exc:
            self.playback.setValue(0)
            self._show_error(str(exc))
            return

        self.state.processed_audio = out_audio
        self.wave_out.set_audio(out_audio)
        self.pitch_graph.set_f0(f0)
        self.metric_latency.setText(f"{metrics['latency_ms']:.0f} ms")
        self.metric_proc.setText(f"{max(0.3, metrics['latency_ms']/220.0):.1f} s")
        self.metric_fid.setText(f"{metrics['quality_pesq_like']:.1f} /5")
        self.metric_int.setText(f"{metrics['intelligibility_mos']:.1f} /5")
        self._append_log(f"[process] {params.mode} / {params.option}")

    def _on_export_audio(self) -> None:
        if self.state.processed_audio.size == 0:
            self._show_error("Process audio first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export audio", "", "WAV (*.wav);;MP3 (*.mp3);;FLAC (*.flac)"
        )
        if not path:
            return
        fmt = Path(path).suffix.replace(".", "").lower() or "wav"
        try:
            output = self.backend.export_audio(path, fmt)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self._append_log(f"[export] audio: {output.name}")

    def _on_export_embedding(self) -> None:
        if self.state.processed_audio.size == 0:
            self._show_error("Process audio first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export embedding", "", "NumPy (*.npy)")
        if not path:
            return
        try:
            output = self.backend.export_embedding(path)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self._append_log(f"[export] embedding: {output.name}")

    def _on_save_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save session log", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.backend.save_log(path)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self._append_log(f"[log] saved: {Path(path).name}")

    def _append_log(self, text: str) -> None:
        self.log_box.append(f"- {text}")

    def _show_error(self, message: str) -> None:
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("OmniSpeech Error")
        msg.setText(message)
        msg.exec()


def run_app() -> None:
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.showMaximized()
    app.exec()
