from abc import ABC, abstractmethod
from typing import List
from core.models import Word, Turn


class ITranscriber(ABC):
    @abstractmethod
    def transcribe(self, path: str) -> List[Word]: ...


class IDiarizer(ABC):
    @abstractmethod
    def diarize(self, path: str) -> List[Turn]: ...
