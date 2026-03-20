"""
tad_demo/app.py
----------------
FastAPI demo server for trublib TAD pipeline.

Endpoints
~~~~~~~~~
POST /process   — upload audio file, returns processed WAV + per-chunk stats
GET  /health    — liveness check
GET  /          — serves the UI

Usage
~~~~~
    pip install fastapi uvicorn python-multipart
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import base64
import io
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from scipy.io import wavfile

from trublib.config import TADConfig
from trublib.processor import TADProcessor

app = FastAPI(title="trublib TAD Demo")

SR = 24_000
CHUNK = 1_920   # 80ms @ 24kHz

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def convert_to_wav(src_bytes: bytes, src_suffix: str) -> np.ndarray:
    """Convert any audio format to 24kHz mono float32 via ffmpeg."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / f"input{src_suffix}"
        dst_path = Path(tmp) / "output.wav"

        src_path.write_bytes(src_bytes)

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(src_path),
                "-ac", "1",
                "-ar", str(SR),
                "-sample_fmt", "s16",
                str(dst_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise ValueError(
                f"ffmpeg conversion failed: {result.stderr.decode()[-300:]}"
            )

        file_sr, data = wavfile.read(str(dst_path))
        if data.dtype == np.int16:
            samples = data.astype(np.float32) / 32768.0
        else:
            samples = data.astype(np.float32)

        return samples


def samples_to_wav_bytes(samples: np.ndarray, sr: int = SR) -> bytes:
    """Convert float32 samples to 16-bit PCM WAV bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    int16 = (clipped * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, int16)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def process_audio(
        samples: np.ndarray,
        threshold: float = 0.6,
        onset_chunks: int = 3,
        trailing_chunks: int = 4,
) -> dict:
    """
    Run the full TAD pipeline over audio samples.

    Returns a dict with:
        output_samples   : np.ndarray — masked audio
        chunks_meta      : list[dict] — per-chunk state, confidence, flush
        stats            : dict — summary statistics
    """
    cfg = TADConfig(
        input_sample_rate=SR,
        threshold=threshold,
        onset_chunks=onset_chunks,
        trailing_chunks=trailing_chunks,
    )
    tad = TADProcessor(cfg)

    output_parts: list[np.ndarray] = []
    chunks_meta: list[dict] = []

    total_chunks = 0
    active_chunks = 0

    for start in range(0, len(samples) - CHUNK, CHUNK):
        chunk = samples[start : start + CHUNK]
        result = tad.process(chunk)

        total_chunks += 1
        if result.is_trumpet:
            active_chunks += 1

        # Retroactive flush: insert attack frames at their correct temporal
        # position in the output stream, immediately before this chunk's audio.
        # Old code collected all flush_samples then prepended to the entire
        # output — placing note-attack frames at t=0 before any silence.
        # Correct: append them here so they land at the moment ACTIVE fires.
        if result.flush:
            for frame in result.flush:
                output_parts.append(frame)

        output_parts.append(result.masked_audio)
        chunks_meta.append({
            "chunk_index": total_chunks - 1,
            "time_sec": round(start / SR, 3),
            "state": result.state.value,
            "confidence": round(float(result.confidence), 4),
            "is_trumpet": result.is_trumpet,
            "had_flush": result.flush is not None,
        })

    # Assemble output
    if output_parts:
        output_samples = np.concatenate(output_parts)
    else:
        output_samples = np.zeros(0, dtype=np.float32)

    stats = {
        "total_chunks": total_chunks,
        "active_chunks": active_chunks,
        "trumpet_ratio": round(active_chunks / total_chunks, 3) if total_chunks else 0,
        "duration_sec": round(len(samples) / SR, 2),
        "output_duration_sec": round(len(output_samples) / SR, 2),
        "state_transitions": _count_transitions(chunks_meta),
    }

    return {
        "output_samples": output_samples,
        "chunks_meta": chunks_meta,
        "stats": stats,
    }


def _count_transitions(meta: list[dict]) -> list[dict]:
    """Extract state transition events for timeline display."""
    transitions = []
    prev_state = None
    for m in meta:
        if m["state"] != prev_state:
            transitions.append({
                "time_sec": m["time_sec"],
                "from": prev_state,
                "to": m["state"],
            })
            prev_state = m["state"]
    return transitions


# ---------------------------------------------------------------------------
# Debug endpoint — raw confidence scores per chunk, no masking
# ---------------------------------------------------------------------------


@app.post("/debug")
async def debug_endpoint(
        file: UploadFile = File(...),
        threshold: float = Form(0.6),
):
    """
    Returns per-chunk confidence scores without applying any masking.
    Use this to diagnose what the model actually scores for your audio.
    """
    suffix = Path(file.filename).suffix.lower() if file.filename else ".wav"
    raw_bytes = await file.read()

    try:
        samples = convert_to_wav(raw_bytes, suffix)
    except Exception as e:
        raise HTTPException(422, f"Audio conversion failed: {e}")

    from trublib.config import TADConfig
    from trublib.feature_extractor import FeatureExtractor
    from trublib.frame_manager import FrameManager
    from trublib.normalizer import TwoStageNormalizer
    from trublib.trumpet_scorer import TrumpetScorer

    cfg    = TADConfig(input_sample_rate=SR, threshold=threshold)
    norm   = TwoStageNormalizer(target_rms=cfg.target_rms)
    fm     = FrameManager()
    ex     = FeatureExtractor(sr=SR)
    scorer = TrumpetScorer()

    chunk_scores = []
    for start in range(0, len(samples) - CHUNK, CHUNK):
        chunk = samples[start : start + CHUNK]
        normalised, rms_db = norm.process(chunk)
        frames = fm.push(normalised)
        fvs = [ex.extract(f, rms_db_override=rms_db) for f in frames]
        confidence = scorer.score(fvs)
        chunk_scores.append({
            "chunk": len(chunk_scores),
            "time_sec": round(start / SR, 3),
            "confidence": round(float(confidence), 4),
            "above_threshold": confidence >= threshold,
            "n_frames": len(fvs),
        })

    above = sum(1 for c in chunk_scores if c["above_threshold"])
    return JSONResponse({
        "total_chunks": len(chunk_scores),
        "above_threshold": above,
        "threshold": threshold,
        "max_confidence": round(max(c["confidence"] for c in chunk_scores), 4),
        "mean_confidence": round(sum(c["confidence"] for c in chunk_scores) / len(chunk_scores), 4),
        "chunks": chunk_scores,
    })


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "sr": SR, "chunk_ms": 80}


@app.post("/process")
async def process_endpoint(
        file: UploadFile = File(...),
        threshold: float = Form(0.6),
        onset_chunks: int = Form(3),
        trailing_chunks: int = Form(4),
):
    # Validate
    if threshold < 0.0 or threshold > 1.0:
        raise HTTPException(400, "threshold must be in [0.0, 1.0]")
    if onset_chunks < 1 or onset_chunks > 10:
        raise HTTPException(400, "onset_chunks must be in [1, 10]")
    if trailing_chunks < 1 or trailing_chunks > 20:
        raise HTTPException(400, "trailing_chunks must be in [1, 20]")

    suffix = Path(file.filename).suffix.lower() if file.filename else ".wav"
    raw_bytes = await file.read()

    if len(raw_bytes) == 0:
        raise HTTPException(400, "Empty file uploaded")
    if len(raw_bytes) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 50MB)")

    try:
        samples = convert_to_wav(raw_bytes, suffix)
    except Exception as e:
        raise HTTPException(422, f"Audio conversion failed: {e}")

    if len(samples) < CHUNK:
        raise HTTPException(422, "Audio too short (minimum 80ms required)")

    try:
        result = process_audio(
            samples,
            threshold=threshold,
            onset_chunks=onset_chunks,
            trailing_chunks=trailing_chunks,
        )
    except Exception:
        raise HTTPException(500, f"Processing failed:\n{traceback.format_exc()}")

    # Encode output WAV as base64
    output_wav = samples_to_wav_bytes(result["output_samples"])
    output_b64 = base64.b64encode(output_wav).decode()

    # Also encode original for comparison playback
    original_wav = samples_to_wav_bytes(samples[: len(result["output_samples"])])
    original_b64 = base64.b64encode(original_wav).decode()

    return JSONResponse({
        "output_audio_b64": output_b64,
        "original_audio_b64": original_b64,
        "chunks_meta": result["chunks_meta"],
        "stats": result["stats"],
    })


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def ui():
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>UI not found — place index.html in static/</h1>", 404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# Serve static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")