#!/usr/bin/env python3
"""Gradio interface for BackTrack4Drum — remove drums from audio using Demucs.

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

# Load model once at startup
print("Loading Demucs model (htdemucs)...")
MODEL = get_model("htdemucs")
MODEL.eval()


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


def remove_drums(audio_path):
    """Separate sources on CPU, return drumless WAV."""
    if audio_path is None:
        raise gr.Error("Please upload an audio file.")

    wav = load_audio(audio_path, MODEL.samplerate, MODEL.audio_channels)
    wav = wav.unsqueeze(0)  # (1, channels, samples)

    sources = apply_model(MODEL, wav, progress=False)
    sources = sources.squeeze(0)  # (num_sources, channels, samples)

    drums_idx = MODEL.sources.index("drums")
    no_drums = sum(sources[i] for i in range(len(MODEL.sources)) if i != drums_idx)

    no_drums_np = no_drums.cpu().numpy().T  # (samples, channels)

    out_path = tempfile.mktemp(suffix=".wav")
    sf.write(out_path, no_drums_np, MODEL.samplerate)
    return out_path


demo = gr.Interface(
    fn=remove_drums,
    inputs=gr.Audio(type="filepath", label="Upload audio (MP3/WAV/FLAC/...)"),
    outputs=gr.Audio(type="filepath", label="Drumless audio"),
    title="BackTrack4Drum",
    description=(
        "Upload a song and get it back without drums. Powered by Demucs (Meta AI).\n\n"
        "Running on free CPU provided by Hugging Face. Processing takes about 5 minutes per song."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
