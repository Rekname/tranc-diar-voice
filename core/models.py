from dataclasses import dataclass, field
from typing import Dict, List


def default_speaker_name(speaker: str) -> str:
    try:
        return f"Спикер {int(speaker.split('_')[-1]) + 1}"
    except ValueError:
        return speaker


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
        return {"start": round(self.start, 2), "end": round(self.end, 2),
                "speaker": self.speaker, "text": self.text.strip()}


@dataclass
class Result:
    language: str
    segments: List[Segment] = field(default_factory=list)
    speakers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        names: Dict[str, str] = {}
        for s in self.segments:
            if s.speaker not in names:
                names[s.speaker] = self.speakers.get(s.speaker) or default_speaker_name(s.speaker)
        return {"language": self.language, "speakers": names,
                "segments": [s.to_dict() for s in self.segments]}
