import os
import tempfile
import librosa
import soundfile as sf
from interfaces.base import ITranscriber, IDiarizer
from core.models import Word, Turn, Segment, Result
from core.log import stage


def _to_wav16k(path: str, dst: str) -> str:
    audio, _ = librosa.load(path, sr=16000, mono=True)
    sf.write(dst, audio, 16000, subtype="PCM_16")
    return dst


def _speaker_of(word: Word, turns: list[Turn]) -> str:
    mid = (word.start + word.end) / 2
    best, best_dist = None, float("inf")
    for t in turns:
        if t.start <= mid <= t.end:
            return t.speaker
        dist = min(abs(mid - t.start), abs(mid - t.end))
        if dist < best_dist:
            best, best_dist = t.speaker, dist
    return best or "speaker_0"


def _merge(words: list[Word], turns: list[Turn]) -> list[Segment]:
    segments: list[Segment] = []
    for w in words:
        spk = _speaker_of(w, turns)
        if segments and segments[-1].speaker == spk:
            segments[-1].end = w.end
            segments[-1].text += w.text
        else:
            segments.append(Segment(w.start, w.end, spk, w.text))
    return segments


class Pipeline:
    def __init__(self, transcriber: ITranscriber, diarizer: IDiarizer):
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
