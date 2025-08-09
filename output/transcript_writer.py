import json
import os
from typing import List, Dict
from loguru import logger


def save_transcript(outdir: str, base_name: str, segments: List[Dict], full_text: str):
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, f"{base_name}.json")
    txt_path = os.path.join(outdir, f"{base_name}.txt")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"segments": segments, "text": full_text}, f, ensure_ascii=False, indent=2)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    logger.info(f"Saved transcript: {json_path} and {txt_path}")
    return json_path, txt_path
