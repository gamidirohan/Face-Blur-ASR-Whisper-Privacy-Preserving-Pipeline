import argparse
import os
import time
from datetime import datetime
from pathlib import Path

from utils.logger import logger
from video_processor import FaceAnonymizer, extract_audio_to_wav
from transcription.whisper_transcriber import WhisperTranscriber
from output.transcript_writer import save_transcript
from output.video_writer import write_video


def parse_args():
    p = argparse.ArgumentParser("Face Blur + ASR")
    # Make inputs optional; if neither provided, default to processing --input-dir
    p.add_argument("--input", type=str, help="Path to input video file")
    p.add_argument("--webcam", type=int, help="Webcam device index")
    p.add_argument("--input-dir", type=str, default="video data", help="Directory with videos to batch process")
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--blur-method", type=str, default="gaussian", choices=["gaussian", "pixelate"]) 
    p.add_argument("--model", type=str, default="base")
    p.add_argument("--language", type=str, default=None, help="Force language code; leave empty for auto-detect (works for non-English)")
    p.add_argument("--subtitle", action="store_true")
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--display", action="store_true")
    return p.parse_args()


def process_file(input_video: str, outdir: str, anonymizer: FaceAnonymizer, model_name: str, language: str, subtitle: bool):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(input_video).stem
    os.makedirs(outdir, exist_ok=True)

    # 1) Face anonymization (initial pass without subtitles)
    video_out = os.path.join(outdir, f"anonymized_{base}_{ts}.mp4")
    logger.info(f"Processing video for face anonymization: {input_video}")
    write_video(input_video, video_out, anonymizer, subtitle_segments=None, realtime=False)

    # 2) Audio extraction and transcription (auto-detect language if not provided)
    wav_out = os.path.join(outdir, f"audio_{base}_{ts}.wav")
    wav_path = extract_audio_to_wav(input_video, wav_out)

    transcriber = WhisperTranscriber(model_name=model_name, language=language)
    result = transcriber.transcribe(wav_path)
    segments = transcriber.to_segments_json(result)
    text = transcriber.to_plain_text(result)
    json_path, txt_path = save_transcript(outdir, f"transcript_{base}_{ts}", segments, text)

    # 3) Optional subtitles overlay
    if subtitle:
        logger.info("Adding subtitles to anonymized video...")
        video_out_sub = os.path.join(outdir, f"anonymized_sub_{base}_{ts}.mp4")
        write_video(input_video, video_out_sub, anonymizer, subtitle_segments=segments, realtime=False)
        logger.info(f"Saved subtitled video: {video_out_sub}")

    logger.info(f"Done: {input_video}")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    anonymizer = FaceAnonymizer(blur_method=args.blur_method)

    # Webcam realtime mode (no ASR by default)
    if args.webcam is not None:
        video_out = os.path.join(args.outdir, f"anonymized_webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        write_video(
            input_path=None,
            out_path=video_out,
            anonymizer=anonymizer,
            subtitle_segments=None,
            realtime=True,
            webcam_index=args.webcam,
            display=args.display,
        )
        logger.info(f"Webcam anonymization saved to {video_out}")
        return

    # Single file mode
    if args.input:
        process_file(args.input, args.outdir, anonymizer, args.model, args.language, args.subtitle)
        return

    # Batch directory mode (default to 'video data')
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        alt = Path("..") / args.input_dir
        if alt.exists() and alt.is_dir():
            logger.info(f"Input directory not found at '{input_dir}', using sibling '{alt}'")
            input_dir = alt
        else:
            logger.error(f"Input directory does not exist: {input_dir}")
            return

    exts = (".mp4", ".mov", ".mkv", ".avi")
    videos = sorted([str(p) for p in input_dir.iterdir() if p.suffix.lower() in exts])
    if not videos:
        logger.warning(f"No videos found in: {input_dir}")
        return

    logger.info(f"Found {len(videos)} videos in '{input_dir}'. Starting batch processing...")
    for vid in videos:
        try:
            process_file(vid, args.outdir, anonymizer, args.model, args.language, args.subtitle)
        except Exception as e:
            logger.exception(f"Failed to process {vid}: {e}")


if __name__ == "__main__":
    main()
