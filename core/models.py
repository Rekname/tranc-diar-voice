from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "speaker": self.speaker,
            "text": self.text.strip(),
        }


@dataclass
class Result:
    language: str
    segments: List[Segment] = field(default_factory=list)
    speakers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        speakers = self.speakers or {s.speaker: _default_name(s.speaker) for s in self.segments}
        seen: Dict[str, str] = {}
        for s in self.segments:
            if s.speaker not in seen:
                seen[s.speaker] = speakers.get(s.speaker, _default_name(s.speaker))
        return {
            "language": self.language,
            "speakers": seen,
            "segments": [s.to_dict() for s in self.segments],
        }


def _default_name(speaker: str) -> str:
    try:
        n = int(speaker.split("_")[-1]) + 1
        return f"Спикер {n}"
    except ValueError:
        return speaker
