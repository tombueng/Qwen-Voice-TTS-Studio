"""Trim leading/trailing silence and normalize volume for every audio file in audio/."""
import subprocess, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

AUDIO_DIR   = Path(__file__).parent / "audio"
THRESHOLD   = 0.003     # ~-50 dB — only strip true silence
PAD_MS      = 80        # keep 80 ms natural decay on each end
NORM_TARGET = 0.891     # -1 dBFS peak  (10^(-1/20))


def _mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1) if data.ndim == 2 else data


def trim(data: np.ndarray, sr: int) -> np.ndarray:
    mono  = _mono(data)
    above = np.where(np.abs(mono) > THRESHOLD)[0]
    if len(above) == 0:
        return data                          # entirely silent — leave untouched
    pad = int(PAD_MS * sr / 1000)
    s   = max(0,         above[0]  - pad)
    e   = min(len(mono), above[-1] + pad + 1)
    return data[s:e] if data.ndim == 1 else data[s:e, :]


def normalize(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return data
    peak = float(np.abs(data).max())
    if peak < 1e-9:
        return data
    return np.clip(data * (NORM_TARGET / peak), -1.0, 1.0).astype(np.float32)


def process_file(path: Path) -> str:
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:
        return f"SKIP  {path.name}  ({e})"

    orig_dur  = len(_mono(data)) / sr
    orig_peak = float(np.abs(data).max()) if data.size else 0.0

    try:
        processed = normalize(trim(data, sr))
    except Exception as e:
        return f"ERR   {path.name}  ({e})"

    new_dur  = len(_mono(processed)) / sr
    new_peak = float(np.abs(processed).max()) if processed.size else 0.0

    tag = f"{orig_dur:6.2f}s -> {new_dur:.2f}s  peak {orig_peak:.3f} -> {new_peak:.3f}"

    if path.suffix.lower() == ".mp3":
        # soundfile can't write MP3 — encode via temp WAV + ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = Path(f.name)
        sf.write(str(tmp_wav), processed, sr)
        tmp_mp3 = path.with_name("__trim_tmp__.mp3")
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_wav),
             "-c:a", "libmp3lame", "-q:a", "2", str(tmp_mp3)],
            capture_output=True,
        )
        tmp_wav.unlink(missing_ok=True)
        if ret.returncode == 0:
            tmp_mp3.replace(path)
            return f"OK    {path.name:46s}  {tag}"
        else:
            tmp_mp3.unlink(missing_ok=True)
            return f"FAIL  {path.name}  ffmpeg: {ret.stderr[-120:].decode(errors='replace')}"
    else:
        # Write via temp WAV + ffmpeg to avoid libsndfile OGG encoder instability
        # on large files.  Always safe: original is never touched until the
        # encode succeeds.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = Path(f.name)
        try:
            sf.write(str(tmp_wav), processed, sr)
        except Exception as e:
            tmp_wav.unlink(missing_ok=True)
            return f"ERR   {path.name}  (wav write failed: {e})"

        ext = path.suffix.lower()
        if ext in (".ogg", ".flac"):
            codec = "libvorbis" if ext == ".ogg" else "flac"
            extra = ["-q:a", "6"] if ext == ".ogg" else []
        else:
            codec, extra = "pcm_s16le", []

        tmp_out = path.with_name(f"__trim_tmp__{path.suffix}")
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_wav), "-c:a", codec] + extra + [str(tmp_out)],
            capture_output=True,
        )
        tmp_wav.unlink(missing_ok=True)
        if ret.returncode == 0:
            tmp_out.replace(path)
            return f"OK    {path.name:46s}  {tag}"
        else:
            tmp_out.unlink(missing_ok=True)
            return f"FAIL  {path.name}  ffmpeg: {ret.stderr[-120:].decode(errors='replace')}"


def main():
    files = sorted(
        p for p in AUDIO_DIR.iterdir()
        if p.suffix.lower() in (".ogg", ".wav", ".flac", ".mp3")
    )
    print(f"Processing {len(files)} file(s) in {AUDIO_DIR}")
    for path in files:
        print(process_file(path), flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
