#!/usr/bin/env python3
"""Gradio interface for BackTrack — remove or extract instruments from audio using Demucs.

Designed for Hugging Face Spaces with ZeroGPU.
"""

import os
import subprocess
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import spaces
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
MODES = ["Remove", "Extract"]


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


@spaces.GPU
def process(audio_path, instrument, mode):
    """Separate sources on GPU, return processed audio."""
    if audio_path is None:
        raise gr.Error("Please upload an audio file.")

    model = get_model("htdemucs_6s")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = INSTRUMENT_MAP[instrument]
    extract = mode == "Extract"

    wav = load_audio(audio_path, model.samplerate, model.audio_channels)
    wav = wav.unsqueeze(0).to(device)  # (1, channels, samples)

    sources = apply_model(model, wav, progress=False)
    sources = sources.squeeze(0)  # (num_sources, channels, samples)

    source_names = model.sources

    if extract:
        result = sources[source_names.index(target)]
    else:
        result = sum(sources[i] for i in range(len(source_names)) if source_names[i] != target)

    result_np = result.cpu().numpy().T  # (samples, channels)

    out_path = tempfile.mktemp(suffix=".wav")
    sf.write(out_path, result_np, model.samplerate)
    return out_path


demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Audio(type="filepath", label="Upload audio (MP3/WAV/FLAC/...)"),
        gr.Dropdown(choices=INSTRUMENTS, value="Drums", label="Instrument"),
        gr.Radio(choices=MODES, value="Remove", label="Mode"),
    ],
    outputs=gr.Audio(type="filepath", label="Processed audio"),
    title="BackTrack",
    description=(
        "Upload a song and remove or extract the selected instrument. Powered by Demucs (Meta AI).\n\n"
        "**Remove** — get the song without the selected instrument (e.g. drumless backing track).\n\n"
        "**Extract** — isolate just the selected instrument (e.g. vocals only).\n\n"
        "Supports: Drums, Guitar, Bass, Vocals, Piano.\n\n"
        "Running on free ZeroGPU provided by Hugging Face. Output format: WAV"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
