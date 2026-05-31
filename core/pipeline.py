import os
import tempfile
from typing import List

import librosa
import soundfile as sf

from core.log import stage
from core.models import Result, Segment, Turn, Word


def _to_wav16k(src: str, dst: str) -> str:
    audio, _ = librosa.load(src, sr=16000, mono=True)
    sf.write(dst, audio, 16000, subtype="PCM_16")
    return dst


def _speaker_of(word: Word, turns: List[Turn]) -> str:
    mid = (word.start + word.end) / 2
    best, best_dist = "speaker_0", float("inf")
    for t in turns:
        if t.start <= mid <= t.end:
            return t.speaker
        dist = min(abs(mid - t.start), abs(mid - t.end))
        if dist < best_dist:
            best, best_dist = t.speaker, dist
    return best


def _merge(words: List[Word], turns: List[Turn]) -> List[Segment]:
    segments: List[Segment] = []
    for w in words:
        spk = _speaker_of(w, turns)
        if segments and segments[-1].speaker == spk:
            segments[-1].end = w.end
            segments[-1].text += w.text
        else:
            segments.append(Segment(w.start, w.end, spk, w.text))
    return segments


class Pipeline:
    def __init__(self, transcriber, diarizer):
        self.transcriber = transcriber
        self.diarizer = diarizer

    def run(self, path: str, on_stage=None) -> Result:
        with tempfile.TemporaryDirectory() as tmp:
            with stage("конвертация WAV 16k", on_stage):
                wav = _to_wav16k(path, os.path.join(tmp, "audio.wav"))
            with stage("транскрипция Whisper", on_stage):
                words = self.transcriber.transcribe(wav)
            with stage("диаризация NeMo", on_stage):
                turns = self.diarizer.diarize(wav)
            with stage("объединение", on_stage):
                segments = _merge(words, turns)
        return Result(language="ru", segments=segments)
