import os
import json
import tempfile
import wave
from typing import List
from core.quiet import muted, silence_nemo
from omegaconf import OmegaConf

with muted():
    from nemo.collections.asr.models import ClusteringDiarizer

from interfaces.base import IDiarizer
from core.models import Turn

silence_nemo()

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "models", "nemo")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "conf", "diar_infer.yaml")
VAD_MODEL = os.path.join(MODEL_DIR, "vad_multilingual_marblenet.nemo")
SPK_MODEL = os.path.join(MODEL_DIR, "titanet_large.nemo")

LEVELS = {
    "easy": {
        "windows": [1.5],
        "shifts": [1.0],
        "weights": [1],
    },
    "medium": {
        "windows": [1.5, 1.25, 1.0, 0.75, 0.5],
        "shifts": [0.75, 0.625, 0.5, 0.375, 0.25],
        "weights": [1, 1, 1, 1, 1],
    },
    "hard": {
        "windows": [2.0, 1.5, 1.25, 1.0, 0.75, 0.5],
        "shifts": [0.5, 0.375, 0.3125, 0.25, 0.1875, 0.125],
        "weights": [1, 1, 1, 1, 1, 1],
    },
}


class NeMoDiarizer(IDiarizer):
    def __init__(self, device: str = "cpu", level: str = "medium"):
        self.device = device
        self.level = level if level in LEVELS else "medium"

    def diarize(self, path: str) -> List[Turn]:
        with wave.open(path, "rb") as w:
            duration = w.getnframes() / w.getframerate()

        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "manifest.json")
            with open(manifest, "w") as f:
                f.write(json.dumps({
                    "audio_filepath": path,
                    "offset": 0,
                    "duration": duration,
                    "label": "infer",
                    "text": "-",
                    "num_speakers": None,
                    "rttm_filepath": None,
                    "uem_filepath": None,
                }))

            preset = LEVELS[self.level]
            cfg = OmegaConf.load(CONFIG_PATH)
            cfg.device = self.device
            cfg.verbose = False
            cfg.diarizer.manifest_filepath = manifest
            cfg.diarizer.out_dir = tmp
            cfg.diarizer.vad.model_path = VAD_MODEL
            cfg.diarizer.speaker_embeddings.model_path = SPK_MODEL
            cfg.diarizer.speaker_embeddings.parameters.save_embeddings = False
            cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = preset["windows"]
            cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec = preset["shifts"]
            cfg.diarizer.speaker_embeddings.parameters.multiscale_weights = preset["weights"]
            cfg.diarizer.clustering.parameters.oracle_num_speakers = False

            with muted():
                ClusteringDiarizer(cfg=cfg).diarize()

            name = os.path.splitext(os.path.basename(path))[0]
            rttm = os.path.join(tmp, "pred_rttms", name + ".rttm")
            return self._parse_rttm(rttm)

    @staticmethod
    def _parse_rttm(path: str) -> List[Turn]:
        turns: List[Turn] = []
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) < 8 or p[0] != "SPEAKER":
                    continue
                start, dur = float(p[3]), float(p[4])
                turns.append(Turn(start, start + dur, p[7]))
        return sorted(turns, key=lambda t: t.start)
