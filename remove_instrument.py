#!/usr/bin/env python3
"""Remove or extract a specified instrument from audio files using Demucs (Meta AI)."""

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

INSTRUMENTS = ("drums", "guitar", "bass", "vocals", "piano")


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


def save_mp3(audio_np, samplerate, out_mp3, bitrate):
    """Save numpy audio array as MP3 via temporary WAV."""
    tmp_wav = out_mp3.replace(".mp3", ".tmp.wav")
    sf.write(tmp_wav, audio_np, samplerate)
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_wav, "-b:a", bitrate, out_mp3],
        capture_output=True,
        check=True,
    )
    os.unlink(tmp_wav)


def process_file(model, filepath, output_dir, bitrate, instrument, extract):
    """Separate sources and save the result."""
    basename = os.path.splitext(os.path.basename(filepath))[0]

    mode_str = "Extracting" if extract else "Removing"
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(filepath)}")
    print(f"{mode_str}: {instrument}")
    print(f"{'='*60}")

    wav = load_audio(filepath, model.samplerate, model.audio_channels)
    wav = wav.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sources = apply_model(model, wav.to(device), progress=True)
    sources = sources.squeeze(0)

    source_names = model.sources
    print(f"Source names: {source_names}")

    if instrument not in source_names:
        raise ValueError(
            f"Instrument '{instrument}' not found in model sources {source_names}. "
            f"Try a model that supports this instrument (e.g. htdemucs_6s for guitar/piano)."
        )

    target_idx = source_names.index(instrument)
    isolated = sources[target_idx].cpu().numpy().T
    backing = sum(sources[i] for i in range(len(source_names)) if i != target_idx).cpu().numpy().T

    if extract:
        out_mp3 = os.path.join(output_dir, f"{basename}_only_{instrument}.mp3")
        save_mp3(isolated, model.samplerate, out_mp3, bitrate)
        print(f"Output: {out_mp3}")
    else:
        out_no = os.path.join(output_dir, f"{basename}_no_{instrument}.mp3")
        out_only = os.path.join(output_dir, f"{basename}_only_{instrument}.mp3")
        save_mp3(backing, model.samplerate, out_no, bitrate)
        save_mp3(isolated, model.samplerate, out_only, bitrate)
        print(f"Output: {out_no}")
        print(f"Output: {out_only}")


def main():
    parser = argparse.ArgumentParser(description="Remove or extract an instrument from audio files using Demucs")
    parser.add_argument("input", nargs="*", help="Input audio files (default: all files in /data/input)")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-r", "--remove", choices=INSTRUMENTS, help="Instrument to remove (default: drums)")
    mode.add_argument("-e", "--extract", choices=INSTRUMENTS, help="Instrument to extract (isolate)")

    parser.add_argument("-o", "--output", default="/data/output", help="Output directory")
    parser.add_argument("-b", "--bitrate", default="48k", help="MP3 bitrate (default: 48k)")
    parser.add_argument("-m", "--model", default="htdemucs_6s", help="Demucs model (default: htdemucs_6s)")
    args = parser.parse_args()

    extract = args.extract is not None
    instrument = args.extract if extract else (args.remove or "drums")

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
        process_file(model, f, args.output, args.bitrate, instrument, extract)

    print(f"\nAll done! Output files are in {args.output}")


if __name__ == "__main__":
    main()
