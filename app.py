#!/usr/bin/env python3
"""Gradio interface for BackTrack — remove or extract instruments from audio using Demucs.

Designed for Hugging Face Spaces (CPU free tier).
"""

import os
import subprocess
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

INSTRUMENTS = ["Drums", "Guitar", "Bass", "Vocals", "Piano"]
INSTRUMENT_MAP = {
    "Drums": "drums",
    "Guitar": "guitar",
    "Bass": "bass",
    "Vocals": "vocals",
    "Piano": "piano",
}

# Load model once at startup
print("Loading Demucs model (htdemucs_6s)...")
MODEL = get_model("htdemucs_6s")
MODEL.eval()

BITRATE = "48k"


def load_audio(path, samplerate, channels):
    """Load audio file via ffmpeg, return torch tensor (channels, samples)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ar", str(samplerate), "-ac", str(channels), tmp_path],
            capture_output=True,
            check=True,
        )
        data, _ = sf.read(tmp_path, dtype="float32")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T
        return torch.from_numpy(data)
    finally:
        os.unlink(tmp_path)


def to_mp3(audio_np, samplerate):
    """Convert numpy audio to MP3 temp file, return path."""
    tmp_wav = tempfile.mktemp(suffix=".wav")
    out_mp3 = tempfile.mktemp(suffix=".mp3")
    sf.write(tmp_wav, audio_np, samplerate)
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_wav, "-b:a", BITRATE, out_mp3],
        capture_output=True,
        check=True,
    )
    os.unlink(tmp_wav)
    return out_mp3


def process(audio_path, instrument):
    """Separate sources on CPU, return backing track and isolated instrument as MP3."""
    if audio_path is None:
        raise gr.Error("Please upload an audio file.")

    target = INSTRUMENT_MAP[instrument]

    wav = load_audio(audio_path, MODEL.samplerate, MODEL.audio_channels)
    wav = wav.unsqueeze(0)  # (1, channels, samples)

    sources = apply_model(MODEL, wav, progress=False)
    sources = sources.squeeze(0)  # (num_sources, channels, samples)

    source_names = MODEL.sources
    target_idx = source_names.index(target)

    backing = sum(sources[i] for i in range(len(source_names)) if i != target_idx).cpu().numpy().T
    isolated = sources[target_idx].cpu().numpy().T

    return to_mp3(backing, MODEL.samplerate), to_mp3(isolated, MODEL.samplerate)


demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Audio(type="filepath", label="Upload audio (MP3/WAV/FLAC/...)"),
        gr.Dropdown(choices=INSTRUMENTS, value="Drums", label="Instrument to remove"),
    ],
    outputs=[
        gr.Audio(type="filepath", label="Backing track (without selected instrument)"),
        gr.Audio(type="filepath", label="Isolated instrument"),
    ],
    title="BackTrack",
    description=(
        "Upload a song and remove the selected instrument. You get two MP3 files back:\n\n"
        "1. **Backing track** — the song without the selected instrument.\n"
        "2. **Isolated instrument** — just the selected instrument by itself.\n\n"
        "Supports: Drums, Guitar, Bass, Vocals, Piano.\n\n"
        "Running on free CPU provided by Hugging Face. Processing takes about 5 minutes per song."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
