import os
from typing import List

from faster_whisper import WhisperModel

from core.models import Word

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "whisper")


class FasterWhisperTranscriber:
    def __init__(self, model_name: str = "small", device: str = "cpu", word_timestamps: bool = False):
        self.word_timestamps = word_timestamps
        self.model = WhisperModel(
            model_name, device=device,
            compute_type="float16" if device == "cuda" else "int8",
            download_root=MODEL_DIR,
        )

    def transcribe(self, path: str) -> List[Word]:
        segments, _ = self.model.transcribe(
            path, language="ru", word_timestamps=self.word_timestamps, vad_filter=True,
        )
        words: List[Word] = []
        for seg in segments:
            if self.word_timestamps and seg.words:
                words.extend(Word(w.start, w.end, w.word) for w in seg.words)
            else:
                words.append(Word(seg.start, seg.end, seg.text))
        return words
