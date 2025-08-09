import os
import subprocess
import shutil
from loguru import logger

try:
    import imageio_ffmpeg  # optional fallback
except Exception:  # pragma: no cover
    imageio_ffmpeg = None


def _resolve_ffmpeg_exe() -> str:
    # 1) Prefer PATH
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # 2) Fallback to imageio-ffmpeg bundled binary
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    raise FileNotFoundError(
        "ffmpeg not found. Install FFmpeg or ensure PATH is refreshed."
    )


def extract_audio_to_wav(video_path: str, out_wav_path: str, sample_rate: int = 16000, overwrite: bool = True) -> str:
    """
    Extract audio from video to mono WAV for ASR.
    Uses system ffmpeg or imageio-ffmpeg fallback.
    """
    if os.path.exists(out_wav_path) and not overwrite:
        logger.info(f"Audio exists: {out_wav_path}")
        return out_wav_path

    ffmpeg_exe = _resolve_ffmpeg_exe()
    logger.info(f"Extracting audio from {video_path} -> {out_wav_path} (ffmpeg: {ffmpeg_exe})")

    cmd = [
        ffmpeg_exe,
        "-y" if overwrite else "-n",
        "-i",
        video_path,
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        out_wav_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr.decode(errors="ignore"))
        raise
    return out_wav_path
