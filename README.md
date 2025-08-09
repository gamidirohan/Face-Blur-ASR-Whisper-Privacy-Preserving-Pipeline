# Face Blur + ASR (Whisper) — Privacy-Preserving Pipeline

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="https://streamlit.io/"><img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white"></a>
  <a href="#license"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="https://github.com/openai/whisper"><img alt="Whisper" src="https://img.shields.io/badge/ASR-Whisper-blue"></a>
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-informational">
</p>

<p align="center">
  <!-- Place a short demo video at demo/demo.mp4 (consider Git LFS). GitHub renders HTML5 video. -->
  <video src="demo/demo.mp4" controls width="720" muted>
    <a href="demo/demo.mp4">View the demo video</a>
  </video>
</p>

Real-time and batch tool to anonymize faces in video and generate captions/transcripts using Whisper.

- Face detection + anonymization (Gaussian blur or pixelation)
- Whisper transcription with non-English auto-detect
- Streamlit live app (webcam) with low-latency captions (chunking + energy gate)
- Batch CLI for folders or single files
- Outputs: anonymized video, transcript JSON/TXT, optional subtitles overlay

## Quickstart

Requirements:
- Windows/macOS/Linux, Python 3.10+
- FFmpeg installed and on PATH (winget install -e --id Gyan.FFmpeg on Windows)

Setup (Windows PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt --python .\.venv\Scripts\python.exe
```

Optional: create a `.env` file
```
WHISPER_MODEL=tiny
ASR_BACKEND=whisper          # whisper | faster-whisper
BLUR_METHOD=gaussian         # gaussian | pixelate
LANGUAGE=                    # leave empty for auto-detect
```

## Streamlit Live App (Webcam)
Run (loads .env if present):
```
Set-Location -LiteralPath "d:\College\Projects\CodeFour AI (comp)\face_blur_asr"; `
if (Test-Path .env) { Get-Content .env | % { if ($_ -and $_ -notmatch '^\s*#') { $n,$v = $_ -split '=',2; if ($n) { $env:$n = $v.Trim('"') } } } }; `
.\.venv\Scripts\streamlit.exe run app.py
```
Features:
- Webcam stream with face blurring in real time
- Low-latency captions via chunked audio + energy-based VAD
- Controls for blur method, ASR model, backend (Whisper / Faster-Whisper)
- Live transcript panel + “Save session transcript”

Recording demo: use Windows Game Bar (Win+G) or OBS to capture the app page. Place the file at `demo/demo.mp4` to render it above.

## CLI (Batch Processing)
Process a folder (defaults to "video data"; auto-detects sibling folder if not found):
```
.\.venv\Scripts\python.exe .\main.py --outdir outputs --model tiny --subtitle
```
Single file:
```
.\.venv\Scripts\python.exe .\main.py --input "video data\1.mp4" --outdir outputs --subtitle
```
Options:
- `--blur-method` gaussian|pixelate
- `--model` tiny|base|small|medium|large
- `--language` language code or leave empty for auto-detect
- `--subtitle` overlay captions on output video

## Outputs
- `outputs/anonymized_<name>_<ts>.mp4`
- `outputs/audio_<name>_<ts>.wav`
- `outputs/transcript_<name>_<ts>.json`
- `outputs/transcript_<name>_<ts>.txt`
- `outputs/anonymized_sub_<name>_<ts>.mp4` (when `--subtitle`)

## Project Structure
```
face_blur_asr/
├── app.py                       # Streamlit live app (webcam blur + captions)
├── main.py                      # CLI pipeline
├── video_processor/
│   ├── face_blur.py             # Haar + blur/pixelate
│   ├── audio_extractor.py       # Robust ffmpeg extraction
├── transcription/
│   ├── whisper_transcriber.py   # Whisper wrapper (Whisper / Faster-Whisper)
├── output/
│   ├── video_writer.py          # Writer + optional subtitle overlay
│   ├── transcript_writer.py     # JSON/TXT writer
├── tests/
│   ├── test_blur.py
│   ├── test_transcriber.py
└── utils/
    └── logger.py
```

## Contributing

Contributions are welcome! Please:
- Open an issue to discuss features/bugs.
- Submit focused PRs with clear descriptions and tests (pytest).
- Keep modules small and documented. Type hints appreciated.

Development:
```
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt --python .\.venv\Scripts\python.exe
.\.venv\Scripts\python.exe -m pytest -q
```

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

## Acknowledgements

- Whisper by OpenAI
- OpenCV for face detection (Haar cascade)
- Streamlit + streamlit-webrtc for real-time UI

## Roadmap
- [ ] SRT/VTT export
- [x] Faster ASR toggle (faster-whisper)
- [x] Better real-time captioning (chunking + energy gate)
- [ ] Dockerfile
