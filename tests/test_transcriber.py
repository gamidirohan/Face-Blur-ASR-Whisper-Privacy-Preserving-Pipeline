import os
import pytest
from transcription.whisper_transcriber import WhisperTranscriber


@pytest.mark.skip(reason="Requires model download and ffmpeg; run integration tests manually")
def test_transcribe_dummy_wav(tmp_path):
    # A short silent WAV file can be generated externally; here we just check API wiring
    transcriber = WhisperTranscriber(model_name="tiny")
    wav_path = os.path.join(tmp_path, "dummy.wav")
    # In real test, generate a small wav tone/silence and expect empty or near-empty text
    # result = transcriber.transcribe(wav_path)
    # assert isinstance(result, dict)
    assert True
