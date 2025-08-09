from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
import os
import json
import torch
import whisper
from loguru import logger
from jiwer import wer
import wave
import numpy as np

try:
    from faster_whisper import WhisperModel as FasterWhisperModel  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FasterWhisperModel = None  # type: ignore


@dataclass
class WhisperTranscriber:
    model_name: str = "base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    language: Optional[str] = None
    backend: Literal["whisper", "faster-whisper"] = "whisper"

    def __post_init__(self):
        logger.info(f"Loading ASR backend={self.backend} model={self.model_name} on {self.device}")
        if self.backend == "faster-whisper":
            if FasterWhisperModel is None:
                logger.warning("faster-whisper not installed; falling back to openai-whisper")
                self.backend = "whisper"
            else:
                compute_type = "int8_float16" if self.device == "cuda" else "int8"
                self.fw_model = FasterWhisperModel(self.model_name, device=self.device, compute_type=compute_type)
        if self.backend == "whisper":
            self.model = whisper.load_model(self.model_name, device=self.device)

    def _load_wav_float32(self, audio_path: str) -> np.ndarray:
        with wave.open(audio_path, 'rb') as w:
            n_channels = w.getnchannels()
            fr = w.getframerate()
            n_frames = w.getnframes()
            sampwidth = w.getsampwidth()
            data = w.readframes(n_frames)
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sampwidth} bytes")
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        if fr != 16000:
            logger.warning(f"Unexpected sample rate {fr}, expected 16000. Consider re-extracting.")
        return audio

    def transcribe(self, audio_path: str) -> Dict:
        logger.info(f"Transcribing: {audio_path}")
        if audio_path.lower().endswith('.wav') and os.path.exists(audio_path):
            audio_arr = self._load_wav_float32(audio_path)
            return self.transcribe_array(audio_arr)
        # Non-wav or path input: defer to backend
        if self.backend == "faster-whisper":
            seg_iter, info = self.fw_model.transcribe(audio_path, language=self.language, task="transcribe")
            segs_list = []
            text_parts = []
            for s in seg_iter:
                segs_list.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})
                text_parts.append(s.text)
            return {"text": "".join(text_parts).strip(), "segments": segs_list}
        else:
            result = self.model.transcribe(audio_path, language=self.language, task="transcribe")
            return result

    def transcribe_array(self, audio: np.ndarray) -> Dict:
        """Transcribe a 1-D float32 numpy array (mono, 16 kHz, -1..1)."""
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if self.backend == "faster-whisper":
            seg_iter, info = self.fw_model.transcribe(audio, language=self.language, task="transcribe")
            segs_list: List[Dict] = []
            text_parts: List[str] = []
            for s in seg_iter:
                segs_list.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})
                text_parts.append(s.text)
            return {"text": "".join(text_parts).strip(), "segments": segs_list}
        else:
            result = self.model.transcribe(audio, language=self.language, task="transcribe")
            return result

    @staticmethod
    def to_segments_json(result: Dict) -> List[Dict]:
        segs = []
        for seg in result.get("segments", []):
            segs.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", "").strip(),
            })
        return segs

    @staticmethod
    def to_plain_text(result: Dict) -> str:
        return result.get("text", "").strip()

    @staticmethod
    def compute_wer(reference: str, hypothesis: str) -> float:
        """Compute Word Error Rate using jiwer."""
        return float(wer(reference, hypothesis))
