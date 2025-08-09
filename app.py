import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from typing import List, Dict, Optional

from video_processor.face_blur import FaceAnonymizer
from transcription.whisper_transcriber import WhisperTranscriber

st.set_page_config(page_title="Face Blur + Live Captions", layout="wide")

@st.cache_resource
def get_anonymizer(method: str):
    return FaceAnonymizer(blur_method=method)

@st.cache_resource
def get_transcriber(model: str, language: Optional[str], backend: str):
    return WhisperTranscriber(model_name=model, language=language, backend=backend)

class BlurTransformer(VideoTransformerBase):
    def __init__(self, anonymizer: FaceAnonymizer):
        self.anonymizer = anonymizer

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        out = self.anonymizer.anonymize(img)
        return out

st.title("Privacy-Preserving Face Blur + Live Captions")

col1, col2 = st.columns([2, 1])
with col2:
    blur_method = st.selectbox("Blur method", ["gaussian", "pixelate"], index=0)
    model = st.selectbox("ASR model", ["tiny", "base", "small", "medium"], index=0)
    backend = st.selectbox("ASR backend", ["whisper", "faster-whisper"], index=0)
    language = st.text_input("Language (leave empty for auto-detect)", value="")
    caption_on = st.toggle("Show live captions", value=True)
    chunk_ms = st.slider("Target chunk length (ms)", 200, 3000, 1200, 100,
                         help="Max chunk size before sending to ASR")
    energy_thresh = st.slider("Energy threshold", 0.001, 0.05, 0.005, 0.001,
                              help="Lower=more sensitive to speech")

anonymizer = get_anonymizer(blur_method)
transcriber = get_transcriber(model, language if language else None, backend)

with col1:
    st.info("Press Start to begin webcam with face blurring. Use the controls for captions.")

# Video only transformer; captions via separate audio track processing
ctx = webrtc_streamer(
    key="blur_captions",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=lambda: BlurTransformer(anonymizer),
    media_stream_constraints={"video": True, "audio": True},
)

# UI elements for captions and transcript
caption_placeholder = st.empty()
transcript_box = st.container(border=True)
save_col1, save_col2 = st.columns([1, 1])

# State to accumulate transcript text
if "session_transcript" not in st.session_state:
    st.session_state.session_transcript = []  # list of strings

if ctx and ctx.state.playing and caption_on:
    if ctx.audio_receiver:
        import queue
        import threading
        import time

        target_rate = 16000
        target_samples = target_rate * chunk_ms // 1000

        voiced_frames: List[np.ndarray] = []
        stop_flag = False

        def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
            if sr == target_rate:
                return x.astype(np.float32, copy=False)
            n = int(round(x.shape[0] * target_rate / sr))
            if n <= 1:
                return x.astype(np.float32, copy=False)
            xp = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=False)
            xq = np.linspace(0.0, 1.0, num=n, endpoint=False)
            y = np.interp(xq, xp, x).astype(np.float32)
            return y

        def audio_worker():
            while not stop_flag:
                try:
                    frame = ctx.audio_receiver.get_frame(timeout=1)
                except queue.Empty:
                    continue
                snd = frame.to_ndarray()
                if snd.ndim > 1:
                    snd = snd.mean(axis=1)
                sr = getattr(frame, "sample_rate", target_rate) or target_rate
                snd = resample_to_16k(snd, int(sr))
                # Simple energy gate
                energy = float(np.mean(snd**2))
                if energy >= energy_thresh:
                    voiced_frames.append(snd)
                # Emit when buffer large enough or slight pause with non-empty buffer
                total = sum(len(v) for v in voiced_frames)
                if total >= target_samples or (voiced_frames and energy < energy_thresh/2):
                    chunk = np.concatenate(voiced_frames)
                    voiced_frames.clear()
                    try:
                        result = transcriber.transcribe_array(chunk)
                        text = result.get("text", "").strip()
                        if text:
                            caption_placeholder.markdown(f"**Caption:** {text}")
                            st.session_state.session_transcript.append(text)
                    except Exception:
                        pass

        t = threading.Thread(target=audio_worker, daemon=True)
        t.start()
        st.write("Live captioning started.")
        st.stop()

# Show running transcript and save button
with transcript_box:
    st.subheader("Session Transcript")
    st.write("\n\n".join(st.session_state.session_transcript[-50:]) or "(no transcript yet)")

with save_col1:
    if st.button("Save session transcript to outputs/"):
        import os, datetime
        os.makedirs("outputs", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = os.path.join("outputs", f"live_transcript_{ts}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(st.session_state.session_transcript))
        st.success(f"Saved: {txt_path}")

st.caption("Tip: Switch ASR backend to 'faster-whisper' for lower latency on CPU/GPU. Adjust energy threshold and chunk length for responsiveness.")
