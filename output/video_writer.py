from typing import Optional, Tuple, List, Dict
import os
import cv2
import numpy as np
from loguru import logger
import shutil
import subprocess

try:
    import imageio_ffmpeg  # optional fallback
except Exception:  # pragma: no cover
    imageio_ffmpeg = None


def _resolve_ffmpeg_exe() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    return None


def _attempt_fix_container(input_path: str) -> Optional[str]:
    ffmpeg_exe = _resolve_ffmpeg_exe()
    if not ffmpeg_exe:
        return None
    base, ext = os.path.splitext(input_path)
    fixed_path = f"{base}_fixed{ext}"
    cmd = [ffmpeg_exe, "-y", "-i", input_path, "-c", "copy", "-movflags", "+faststart", fixed_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(fixed_path):
            logger.warning(f"Input not readable by OpenCV. Remuxed container: {fixed_path}")
            return fixed_path
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr.decode(errors="ignore"))
    return None


def put_subtitle(frame, text: str, pos=(30, 40), font_scale=0.7, color=(255, 255, 255), bg_color=(0, 0, 0)):
    if not text:
        return frame
    overlay = frame.copy()
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    x, y = pos
    cv2.rectangle(overlay, (x-10, y-25), (x + w + 10, y + 10), bg_color, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    return frame


def write_video(
    input_path: Optional[str],
    out_path: str,
    anonymizer,
    subtitle_segments: Optional[List[Dict]] = None,
    realtime: bool = False,
    webcam_index: int = 0,
    display: bool = False,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if realtime:
        cap = cv2.VideoCapture(webcam_index)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            fixed = _attempt_fix_container(input_path)
            if fixed:
                cap = cv2.VideoCapture(fixed)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    seg_idx = 0
    cur_sub = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = anonymizer.anonymize(frame)

        if subtitle_segments:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # Advance segment pointer
            while seg_idx < len(subtitle_segments) and subtitle_segments[seg_idx]["end"] < t:
                seg_idx += 1
            if seg_idx < len(subtitle_segments):
                seg = subtitle_segments[seg_idx]
                if seg["start"] <= t <= seg["end"]:
                    cur_sub = seg["text"]
                else:
                    cur_sub = ""
            else:
                cur_sub = ""
            frame = put_subtitle(frame, cur_sub)

        writer.write(frame)
        if display:
            cv2.imshow("Anonymized", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    logger.info(f"Saved video: {out_path}")
