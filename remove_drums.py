#!/usr/bin/env python3
"""Remove drums from audio files using Demucs (Meta AI)."""

import argparse
import glob
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


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


def process_file(model, filepath, output_dir, bitrate):
    """Separate drums and save the drumless version."""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    out_mp3 = os.path.join(output_dir, f"{basename}_no_drums.mp3")
    tmp_wav = os.path.join(output_dir, f"{basename}_no_drums.wav")

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(filepath)}")
    print(f"{'='*60}")

    wav = load_audio(filepath, model.samplerate, model.audio_channels)
    wav = wav.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sources = apply_model(model, wav.to(device), progress=True)
    sources = sources.squeeze(0)

    drums_idx = model.sources.index("drums")
    no_drums = sum(sources[i] for i in range(len(model.sources)) if i != drums_idx)

    sf.write(tmp_wav, no_drums.cpu().numpy().T, model.samplerate)
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_wav, "-b:a", bitrate, out_mp3],
        capture_output=True,
        check=True,
    )
    os.unlink(tmp_wav)

    print(f"Output: {out_mp3}")


def main():
    parser = argparse.ArgumentParser(description="Remove drums from audio files using Demucs")
    parser.add_argument("input", nargs="*", help="Input audio files (default: all files in /data/input)")
    parser.add_argument("-o", "--output", default="/data/output", help="Output directory")
    parser.add_argument("-b", "--bitrate", default="320k", help="MP3 bitrate (default: 320k)")
    parser.add_argument("-m", "--model", default="htdemucs", help="Demucs model (default: htdemucs)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = args.input
    if not files:
        exts = ("*.mp3", "*.wav", "*.flac", "*.ogg", "*.m4a", "*.wma", "*.aac")
        for ext in exts:
            files.extend(glob.glob(os.path.join("/data/input", ext)))
        files.sort()

    if not files:
        print("No input files found. Place audio files in /data/input or pass them as arguments.")
        return

    print(f"Found {len(files)} file(s)")
    print(f"Loading model: {args.model}")

    model = get_model(args.model)
    model.eval()

    for f in files:
        process_file(model, f, args.output, args.bitrate)

    print(f"\nAll done! Output files are in {args.output}")


if __name__ == "__main__":
    main()
