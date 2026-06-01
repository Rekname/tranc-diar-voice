import json
import os
from html import escape

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
    QStackedWidget, QTextBrowser, QTextEdit, QVBoxLayout, QWidget,
)

from core.models import default_speaker_name
from core.pipeline import Pipeline
from modules.diarizer import NeMoDiarizer
from modules.transcriber import FasterWhisperTranscriber

MODELS = ["tiny", "base", "small", "medium", "large-v3"]
DIAR_LEVELS = [("easy", "Лёгкая (быстро)"), ("medium", "Средняя"), ("hard", "Тяжёлая (точно)")]
DEFAULT_MODEL = "small"
DEFAULT_DIAR_LEVEL = "medium"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
SPEAKER_COLORS = ["#4a90d9", "#e08a3c", "#3eaa6e", "#c25a91",
                  "#7d5cb8", "#bfa628", "#3aa1a9", "#c85a4a"]
CARD_BG = "#ffffff"
CARD_BORDER = "#dcdcdc"
TEXT_PRIMARY = "#1a1a1a"
TEXT_SECONDARY = "#666666"

CARD_STYLE = f"SegmentCard{{background:{CARD_BG};border:1px solid {CARD_BORDER};}}"
EDIT_STYLE = f"QTextEdit{{background:#ffffff;color:{TEXT_PRIMARY};border:none;}}"
LINE_STYLE = (f"QLineEdit{{background:#ffffff;color:{TEXT_PRIMARY};"
              f"border:1px solid {CARD_BORDER};padding:4px 6px;}}")
BROWSER_STYLE = (f"QTextBrowser{{background:#f5f5f5;color:{TEXT_PRIMARY};"
                 "border:1px solid #d0d0d0;padding:8px;}")
SCROLL_STYLE = "QScrollArea{background:#f5f5f5;border:1px solid #d0d0d0;}"


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _speaker_color(speaker: str) -> str:
    try:
        idx = int(speaker.split("_")[-1])
    except ValueError:
        idx = abs(hash(speaker))
    return SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]


def _header_html(label: str, color: str, start: float, end: float) -> str:
    return (f'<span style="color:{color};font-weight:bold">{escape(label)}</span>'
            f'<span style="color:{TEXT_SECONDARY};margin-left:8px">[{start:.1f}–{end:.1f}s]</span>')


class Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str, str, float)

    def __init__(self, pipeline: Pipeline, path: str):
        super().__init__()
        self.pipeline, self.path = pipeline, path
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            result = self.pipeline.run(self.path, on_stage=self.progress.emit)
            if not self.cancelled:
                self.done.emit(result)
        except Exception as e:
            if not self.cancelled:
                self.failed.emit(str(e))


class SegmentCard(QFrame):
    def __init__(self, segment: dict):
        super().__init__()
        self.segment = segment
        self.speaker = segment["speaker"]
        self.color = _speaker_color(self.speaker)
        self.setStyleSheet(CARD_STYLE)

        stripe = QFrame()
        stripe.setFixedWidth(6)
        stripe.setStyleSheet(f"background:{self.color};")

        self.header = QLabel()
        self.header.setTextFormat(Qt.TextFormat.RichText)

        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(segment.get("text", "").strip())
        self.text_edit.setStyleSheet(EDIT_STYLE)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.text_edit.document().contentsChanged.connect(self._fit_height)
        self.text_edit.document().documentLayout().documentSizeChanged.connect(
            lambda _: self._fit_height())

        inner = QVBoxLayout()
        inner.setContentsMargins(10, 8, 10, 8)
        inner.addWidget(self.header)
        inner.addWidget(self.text_edit)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(stripe)
        row.addLayout(inner, 1)

    def set_label(self, label: str):
        self.header.setText(_header_html(label, self.color, self.segment["start"], self.segment["end"]))

    def updated_text(self) -> str:
        return self.text_edit.toPlainText().strip()

    def _fit_height(self):
        doc = self.text_edit.document()
        doc.setTextWidth(self.text_edit.viewport().width())
        self.text_edit.setFixedHeight(max(28, int(doc.size().height()) + 8))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_height()


class JsonEditor(QScrollArea):
    edited = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setStyleSheet(SCROLL_STYLE)
        self._inner = QWidget()
        self._inner.setStyleSheet("background:#f5f5f5;")
        self._layout = QVBoxLayout(self._inner)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(8)
        self.setWidget(self._inner)
        self.cards: list[SegmentCard] = []
        self.speakers: dict[str, str] = {}
        self.data: dict = {}
        self.source_path: str = ""

    def load(self, data: dict, source_path: str):
        self.data = data
        self.source_path = source_path
        self._clear()

        segs = data.get("segments", [])
        stored = data.get("speakers") or {}
        order: list[str] = []
        for s in segs:
            if s["speaker"] not in order:
                order.append(s["speaker"])
        self.speakers = {spk: stored.get(spk, default_speaker_name(spk)) for spk in order}

        self._layout.addWidget(self._build_names_panel())
        for seg in segs:
            card = SegmentCard(seg)
            card.set_label(self.speakers[card.speaker])
            card.text_edit.textChanged.connect(self.edited)
            self.cards.append(card)
            self._layout.addWidget(card)
        self._layout.addStretch(1)

    def _build_names_panel(self) -> QFrame:
        panel = QFrame()
        panel.setStyleSheet(f"QFrame{{background:{CARD_BG};border:1px solid {CARD_BORDER};}}")
        grid = QGridLayout(panel)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        for i, spk in enumerate(self.speakers):
            dot = QLabel()
            dot.setFixedSize(12, 12)
            dot.setStyleSheet(f"background:{_speaker_color(spk)};border-radius:6px;")
            edit = QLineEdit(self.speakers[spk])
            edit.setStyleSheet(LINE_STYLE)
            edit.textChanged.connect(lambda text, s=spk: self._on_name_changed(s, text))
            grid.addWidget(dot, i, 0)
            grid.addWidget(edit, i, 1)
        grid.setColumnStretch(1, 1)
        return panel

    def _on_name_changed(self, speaker: str, name: str):
        self.speakers[speaker] = name
        label = name.strip() or default_speaker_name(speaker)
        for card in self.cards:
            if card.speaker == speaker:
                card.set_label(label)
        self.edited.emit()

    def collect(self) -> dict:
        for card in self.cards:
            card.segment["text"] = card.updated_text()
        speakers = {spk: (name.strip() or default_speaker_name(spk))
                    for spk, name in self.speakers.items()}
        self.data = {"language": self.data.get("language", "ru"),
                     "speakers": speakers,
                     "segments": self.data.get("segments", [])}
        return self.data

    def _clear(self):
        while self._layout.count():
            w = self._layout.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self.cards = []
        self.speakers = {}


class SettingsPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedWidth(260)

        self.model = QComboBox()
        self.model.addItems(MODELS)
        self.model.setCurrentText(DEFAULT_MODEL)

        self.gpu = QCheckBox("включить")
        if not _cuda_available():
            self.gpu.setEnabled(False)
            self.gpu.setToolTip("CUDA недоступна — нет GPU NVIDIA или не установлены драйверы")

        self.word_ts = QCheckBox("точные таймстампы слов")
        self.word_ts.setToolTip("Сильно замедляет транскрипцию (~×2).\n"
                                "Без неё границы между спикерами огрубляются (~2-5 сек).")

        self.diar = QComboBox()
        for key, label in DIAR_LEVELS:
            self.diar.addItem(label, userData=key)
        self.diar.setCurrentIndex(
            next(i for i, (k, _) in enumerate(DIAR_LEVELS) if k == DEFAULT_DIAR_LEVEL))
        self.diar.setToolTip("Лёгкая — 1 окно, быстро\n"
                             "Средняя — 5 окон, баланс\n"
                             "Тяжёлая — 6 окон, максимум точности")

        form = QFormLayout(self)
        form.addRow(QLabel("<b>Настройки</b>"))
        form.addRow("Модель:", self.model)
        form.addRow("GPU:", self.gpu)
        form.addRow("Диаризация:", self.diar)
        form.addRow(self.word_ts)

    def diar_level(self) -> str:
        return self.diar.currentData()


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.path: str | None = None
        self.transcribers: dict[tuple, FasterWhisperTranscriber] = {}
        self.diarizers: dict[tuple, NeMoDiarizer] = {}
        self.progress_lines: list[str] = []
        self.resize(900, 600)

        self.pick = QPushButton("Файл…")
        self.name = QLabel("—")
        self.toggle = QPushButton("Настройки")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.run = QPushButton("Обработать")
        self.run.setEnabled(False)
        self.cancel = QPushButton("Отмена")
        self.cancel.setEnabled(False)
        self.show_btn = QPushButton("Отобразить")
        self.save_btn = QPushButton("Сохранить")
        self.save_btn.setEnabled(False)

        self.pick.clicked.connect(self._pick)
        self.toggle.toggled.connect(lambda c: self.settings.setVisible(c))
        self.run.clicked.connect(self._run)
        self.cancel.clicked.connect(self._cancel)
        self.show_btn.clicked.connect(self._show_json)
        self.save_btn.clicked.connect(self._save_edits)

        top = QHBoxLayout()
        for w in (self.pick, self.name, self.toggle, self.run,
                  self.cancel, self.show_btn, self.save_btn):
            top.addWidget(w)
        top.setStretch(1, 1)

        self.out = QTextBrowser()
        self.out.setOpenExternalLinks(False)
        self.out.setStyleSheet(BROWSER_STYLE)

        self.editor = JsonEditor()
        self.editor.edited.connect(self._on_edited)

        self.viewport = QStackedWidget()
        self.viewport.addWidget(self.out)
        self.viewport.addWidget(self.editor)
        self.settings = SettingsPanel()

        body = QHBoxLayout()
        body.addWidget(self.viewport, 1)
        body.addWidget(self.settings)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(body)

    def _pick(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Аудиофайл", "", "Audio (*.wav *.mp3 *.m4a *.ogg *.flac)")
        if path:
            self.path = path
            self.name.setText(path.rsplit("/", 1)[-1])
            self.run.setEnabled(True)

    def _pipeline(self) -> Pipeline:
        device = "cuda" if self.settings.gpu.isChecked() else "cpu"
        name = self.settings.model.currentText()
        word_ts = self.settings.word_ts.isChecked()
        level = self.settings.diar_level()
        key = (name, device, word_ts)
        if key not in self.transcribers:
            self.out.setPlainText(f"Загрузка {name} ({device})…")
            QApplication.processEvents()
            self.transcribers[key] = FasterWhisperTranscriber(name, device, word_ts)
        diar_key = (device, level)
        if diar_key not in self.diarizers:
            self.diarizers[diar_key] = NeMoDiarizer(device, level)
        return Pipeline(self.transcribers[key], self.diarizers[diar_key])

    def _run(self):
        self.run.setEnabled(False)
        self.cancel.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.progress_lines = []
        self.out.setPlainText("")
        self.viewport.setCurrentWidget(self.out)
        self.worker = Worker(self._pipeline(), self.path)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _on_progress(self, name: str, status: str, dt: float):
        if status == "start":
            self.progress_lines.append(f"{name}…")
        else:
            self.progress_lines[-1] = f"{name} — {dt:.1f}s ({dt / 60:.1f} мин)"
        self.out.setHtml(self._render_progress())

    def _cancel(self):
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.cancel()
        self.out.setPlainText("Отменено")
        self.cancel.setEnabled(False)
        self.run.setEnabled(True)

    def _done(self, result):
        self._save_json(result)
        self.out.setHtml(self._render_progress() + self._render_segments(result))
        self.cancel.setEnabled(False)
        self.run.setEnabled(True)

    def _failed(self, msg: str):
        self.out.setPlainText(f"Ошибка: {msg}")
        self.cancel.setEnabled(False)
        self.run.setEnabled(True)

    def _save_json(self, result):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.path))[0]
        with open(os.path.join(RESULTS_DIR, f"{base}.json"), "w") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    def _render_progress(self) -> str:
        items = "".join(f"<div>{escape(line)}</div>" for line in self.progress_lines)
        return (f'<div style="color:{TEXT_SECONDARY};font-family:monospace;'
                f'margin-bottom:12px">{items}</div>')

    def _render_segments(self, result) -> str:
        if hasattr(result, "segments"):
            segs = [{"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text}
                    for s in result.segments]
            names = getattr(result, "speakers", {}) or {}
        else:
            segs = result.get("segments", [])
            names = result.get("speakers") or {}

        rows = []
        for seg in segs:
            spk = seg["speaker"]
            color = _speaker_color(spk)
            label = names.get(spk) or default_speaker_name(spk)
            time = f"{seg['start']:.1f}–{seg['end']:.1f}s"
            rows.append(
                f'<table cellspacing="0" cellpadding="0" width="100%" '
                f'style="margin:8px 0;background:{CARD_BG};border:1px solid {CARD_BORDER}">'
                f'<tr><td width="6" style="background:{color}"></td>'
                f'<td style="padding:10px 14px;color:{TEXT_PRIMARY}">'
                f'<div style="margin-bottom:4px">'
                f'<span style="color:{color};font-weight:bold">{escape(label)}</span>'
                f'<span style="color:{TEXT_SECONDARY};margin-left:8px">[{time}]</span></div>'
                f'<div style="color:{TEXT_PRIMARY};line-height:1.5">{escape(seg["text"].strip())}</div>'
                f'</td></tr></table>')
        return "".join(rows)

    def _show_json(self):
        start_dir = RESULTS_DIR if os.path.isdir(RESULTS_DIR) else ""
        path, _ = QFileDialog.getOpenFileName(self, "Открыть JSON", start_dir, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self.editor.load(data, path)
            self.viewport.setCurrentWidget(self.editor)
            self.save_btn.setEnabled(True)
            self.name.setText(os.path.basename(path))
        except Exception as e:
            self.out.setPlainText(f"Не удалось открыть JSON: {e}")
            self.viewport.setCurrentWidget(self.out)

    def _save_edits(self):
        if not self.editor.source_path:
            return
        try:
            with open(self.editor.source_path, "w") as f:
                json.dump(self.editor.collect(), f, ensure_ascii=False, indent=2)
            self.save_btn.setText("Сохранено")
            self.save_btn.setEnabled(False)
        except Exception as e:
            self.out.setPlainText(f"Не удалось сохранить: {e}")
            self.viewport.setCurrentWidget(self.out)

    def _on_edited(self):
        self.save_btn.setText("Сохранить")
        self.save_btn.setEnabled(True)


def create_ui():
    app = QApplication.instance() or QApplication([])
    return app, Window()
