#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["whisperx>=3.8", "huggingface_hub>=0.25"]
# ///
"""
transcribe.py – one-shot transcription of a meeting recording with speaker labels.

    uv run transcribe.py <video|audio|sharepoint-url> [options]

Stages (each is skipped when its output already exists in the work dir):
  1. download   – SharePoint anonymous share link or any direct URL
  2. audio      – ffmpeg -> mono 16 kHz WAV
  3. transcribe – WhisperX (default large-v3), word-level alignment
  4. diarize    – pyannote via WhisperX (needs a HuggingFace token)
  5. name       – interactive: you tell the script who SPEAKER_XX is
  6. write      – <name>.txt and <name>.srt with speaker labels

Speaker mapping is stored in <workdir>/speakers.json; re-run with --rename to
change it without touching the expensive stages.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# ----------------------------------------------------------------------------- utils

def log(msg: str) -> None:
    print(f"\033[1;34m»\033[0m {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"\033[1;31m✗\033[0m {msg}", file=sys.stderr)
    sys.exit(code)


def fmt_ts(t: float, sep: str = ",") -> str:
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", sep)


def fmt_min(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


# ----------------------------------------------------------------------------- stage 1: download

SHAREPOINT_RE = re.compile(r"https://([^/]+)/:[a-z]:/g/personal/([^/]+)/([A-Za-z0-9_-]+)")


def resolve_download_url(url: str) -> str:
    """Turn a SharePoint 'share' link into a direct download URL, else return as-is."""
    m = SHAREPOINT_RE.match(url)
    if m:
        host, user, share_id = m.groups()
        return f"https://{host}/personal/{user}/_layouts/15/download.aspx?share={share_id}"
    return url


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"download: using existing {dest.name}")
        return dest
    direct = resolve_download_url(url)
    log(f"download: {direct}")
    if shutil.which("curl"):
        cj = dest.with_suffix(".cookies")
        cmd = ["curl", "-sS", "-L", "-c", str(cj), "-b", str(cj), "-o", str(dest),
               "-w", "%{http_code} %{content_type}\n", direct]
        out = subprocess.run(cmd, capture_output=True, text=True)
        cj.unlink(missing_ok=True)
        status = out.stdout.strip()
        if out.returncode != 0 or not status.startswith("200"):
            dest.unlink(missing_ok=True)
            die(f"download failed ({status or out.stderr.strip()}). "
                "If the link requires a login, download the file in your browser and pass the path instead.")
        if "text/html" in status:
            dest.unlink(missing_ok=True)
            die("download returned an HTML page instead of media – the link probably requires a login. "
                "Download it in your browser and pass the file path instead.")
    else:
        with urllib.request.urlopen(direct) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    log(f"download: {dest.stat().st_size / 1e6:.0f} MB")
    return dest


# ----------------------------------------------------------------------------- stage 2: audio

def extract_audio(src: Path, wav: Path) -> Path:
    if wav.exists():
        log(f"audio: using existing {wav.name}")
        return wav
    if not shutil.which("ffmpeg"):
        die("ffmpeg not found – install it (e.g. `sudo pacman -S ffmpeg` / `apt install ffmpeg`).")
    log("audio: extracting mono 16 kHz WAV")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(wav)], check=True)
    return wav


# ----------------------------------------------------------------------------- preflight

PYANNOTE_MODELS = ["pyannote/speaker-diarization-community-1", "pyannote/segmentation-3.0"]


def preflight(diarize: bool, device: str) -> None:
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            die("CUDA not available. Pass --device cpu (slow) or fix your GPU setup.")
        free, total = torch.cuda.mem_get_info()
        log(f"gpu: {torch.cuda.get_device_name(0)}, {free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
    if not diarize:
        return
    from huggingface_hub import get_token, auth_check
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    if not get_token():
        die("No HuggingFace token found. Speaker diarization needs one:\n"
            "  1. create a Read token at https://huggingface.co/settings/tokens\n"
            "  2. run:  python -c 'from huggingface_hub import login; login()'\n"
            "or pass --no-diarize to skip speaker labels.")
    for repo in PYANNOTE_MODELS:
        try:
            auth_check(repo)
        except GatedRepoError:
            die(f"No access to gated model {repo}.\n"
                f"  Open https://huggingface.co/{repo} and click 'Agree and access repository', then re-run.")
        except RepositoryNotFoundError:
            die(f"Model {repo} not found (renamed?). Check https://huggingface.co/pyannote")
    log("huggingface: token OK, pyannote models accessible")


# ----------------------------------------------------------------------------- stage 3: transcribe

def transcribe(wav: Path, out_json: Path, model_name: str, language: str | None,
               device: str, batch_size: int) -> dict:
    if out_json.exists():
        log(f"transcribe: using existing {out_json.name}")
        return json.load(open(out_json))
    import whisperx
    import torch
    compute = "float16" if device == "cuda" else "int8"
    log(f"transcribe: loading {model_name} ({compute}) on {device}")
    model = whisperx.load_model(model_name, device, compute_type=compute, language=language)
    audio = whisperx.load_audio(str(wav))
    log(f"transcribe: {len(audio) / 16000 / 60:.0f} min of audio, batch_size={batch_size}")
    while True:
        try:
            result = model.transcribe(audio, batch_size=batch_size, language=language, print_progress=True)
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                batch_size //= 2
                log(f"transcribe: GPU out of memory, retrying with batch_size={batch_size}")
                torch.cuda.empty_cache()
                continue
            raise
    lang = result.get("language", language)
    log(f"transcribe: aligning words (language={lang})")
    align_model, meta = whisperx.load_align_model(language_code=lang, device=device)
    result = whisperx.align(result["segments"], align_model, meta, audio, device, return_char_alignments=False)
    result["language"] = lang
    del model, align_model
    if device == "cuda":
        torch.cuda.empty_cache()
    json.dump(result, open(out_json, "w"), ensure_ascii=False)
    return result


# ----------------------------------------------------------------------------- stage 4: diarize

def diarize(wav: Path, result: dict, out_json: Path, device: str) -> dict:
    if out_json.exists():
        log(f"diarize: using existing {out_json.name}")
        return json.load(open(out_json))
    import whisperx
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers
    log("diarize: running pyannote (this takes a few minutes)")
    audio = whisperx.load_audio(str(wav))
    segs = DiarizationPipeline(device=device)(audio)
    result = assign_word_speakers(segs, result)
    json.dump(result, open(out_json, "w"), ensure_ascii=False)
    return result


# ----------------------------------------------------------------------------- stage 5: name speakers

def speaker_stats(result: dict) -> list[dict]:
    stats: dict[str, dict] = {}
    for seg in result["segments"]:
        spk = seg.get("speaker", "UNKNOWN")
        st = stats.setdefault(spk, {"id": spk, "time": 0.0, "first": seg["start"], "last": seg["end"], "segs": []})
        st["time"] += seg["end"] - seg["start"]
        st["last"] = seg["end"]
        st["segs"].append(seg)
    return sorted(stats.values(), key=lambda s: -s["time"])


def ask_names(result: dict, cfg_path: Path, default_label: str, top: int) -> dict:
    stats = speaker_stats(result)
    total = sum(s["time"] for s in stats) or 1
    print()
    print("=" * 72)
    print(f" {len(stats)} speakers found. Name the {min(top, len(stats))} who talked most;")
    print(f" everyone else gets the default label. Enter = default, '=' = same as previous.")
    print("=" * 72)
    mapping: dict[str, str] = {}
    prev = None
    for s in stats[:top]:
        print(f"\n\033[1m{s['id']}\033[0m  {s['time'] / 60:.1f} min ({100 * s['time'] / total:.0f}%), "
              f"{fmt_min(s['first'])} – {fmt_min(s['last'])}")
        samples = sorted(s["segs"], key=lambda x: -(x["end"] - x["start"]))[:3]
        for seg in sorted(samples, key=lambda x: x["start"]):
            txt = seg["text"].strip()
            print(f"   [{fmt_min(seg['start'])}] {txt[:160]}{'…' if len(txt) > 160 else ''}")
        while True:
            ans = input(f"   label for {s['id']} [{default_label}]: ").strip()
            if ans == "=" and prev:
                ans = prev
            if ans and not re.fullmatch(r"[A-Za-z0-9_.-]+", ans):
                print("   use letters, digits, _ . - only (it goes into [brackets] and filenames)")
                continue
            break
        if ans:
            mapping[s["id"]] = ans
            prev = ans
    rest = len(stats) - min(top, len(stats))
    default = default_label
    if rest > 0:
        default = input(f"\nlabel for the remaining {rest} speakers [{default_label}]: ").strip() or default_label
    cfg = {"default": default, "speakers": mapping}
    json.dump(cfg, open(cfg_path, "w"), ensure_ascii=False, indent=2)
    log(f"name: saved {cfg_path}")
    return cfg


# ----------------------------------------------------------------------------- stage 6: write

def write_outputs(result: dict, base: Path, cfg: dict | None) -> None:
    mapping = (cfg or {}).get("speakers", {})
    default = (cfg or {}).get("default")
    has_speakers = any("speaker" in s for s in result["segments"])

    def label(seg):
        if not has_speakers:
            return None
        spk = seg.get("speaker", "UNKNOWN")
        return mapping.get(spk, default or spk)

    txt, srt = base.with_suffix(".txt"), base.with_suffix(".srt")
    with open(txt, "w") as f, open(srt, "w") as g:
        last = object()
        for i, seg in enumerate(result["segments"], 1):
            text = seg["text"].strip()
            lb = label(seg)
            if lb is None:
                f.write(text + "\n")
                g.write(f"{i}\n{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}\n{text}\n\n")
                continue
            if lb != last:
                f.write(f"\n\n[{lb}] ")
                last = lb
            f.write(text + " ")
            g.write(f"{i}\n{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}\n[{lb}] {text}\n\n")
    words = sum(len(s["text"].split()) for s in result["segments"])
    log(f"write: {txt}  ({words} words)")
    log(f"write: {srt}")


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="video/audio file, or URL (SharePoint share link or direct)")
    ap.add_argument("-w", "--workdir", help="where to keep intermediate files (default: ./work/<name>)")
    ap.add_argument("-n", "--name", help="base name of outputs (default: input file stem)")
    ap.add_argument("-m", "--model", default="large-v3")
    ap.add_argument("-l", "--language", default=None, help="e.g. cs, en; default: auto-detect")
    ap.add_argument("-b", "--batch-size", type=int, default=4, help="lower if you run out of GPU memory")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--no-diarize", action="store_true", help="skip speaker labels entirely")
    ap.add_argument("--default-label", default="speaker", help="label for un-named speakers")
    ap.add_argument("--top", type=int, default=6, help="how many speakers to ask about")
    ap.add_argument("--rename", action="store_true", help="only redo the naming step over existing results")
    ap.add_argument("-y", "--yes", action="store_true", help="non-interactive: keep SPEAKER_XX / existing speakers.json")
    a = ap.parse_args()

    is_url = a.input.startswith(("http://", "https://"))
    if is_url:
        parsed = urllib.parse.urlparse(a.input)
        stem = a.name or (Path(parsed.path).stem if not SHAREPOINT_RE.match(a.input) else "recording")
    else:
        src = Path(a.input)
        if not src.exists() and not a.rename:
            die(f"input not found: {src}")
        stem = a.name or src.stem
    work = Path(a.workdir) if a.workdir else Path("work") / stem
    work.mkdir(parents=True, exist_ok=True)
    log(f"workdir: {work}")

    wav = work / f"{stem}.wav"
    raw_json = work / f"{stem}.transcript.json"
    dia_json = work / f"{stem}.diarized.json"
    cfg_path = work / "speakers.json"
    out_base = work / stem

    do_diarize = not a.no_diarize

    if a.rename:
        if not dia_json.exists():
            die(f"{dia_json} not found – run without --rename first.")
        result = json.load(open(dia_json))
        cfg = ask_names(result, cfg_path, a.default_label, a.top)
        write_outputs(result, out_base, cfg)
        return

    if not (raw_json.exists() or wav.exists()):
        src = download(a.input, work / f"{stem}.media") if is_url else src  # type: ignore[possibly-undefined]
        extract_audio(src, wav)
    elif not wav.exists():
        pass  # transcript already exists; wav not needed unless diarizing
    if do_diarize and not dia_json.exists() and not wav.exists():
        die("wav missing but needed for diarization – delete the work dir or re-run with the media file.")

    preflight(do_diarize and not dia_json.exists(), a.device)
    result = transcribe(wav, raw_json, a.model, a.language, a.device, a.batch_size)

    cfg = None
    if do_diarize:
        result = diarize(wav, result, dia_json, a.device)
        if cfg_path.exists():
            cfg = json.load(open(cfg_path))
            log(f"name: using existing {cfg_path}")
        elif not a.yes and sys.stdin.isatty():
            cfg = ask_names(result, cfg_path, a.default_label, a.top)
        else:
            log("name: non-interactive, keeping SPEAKER_XX labels (run --rename later)")
    write_outputs(result, out_base, cfg)


if __name__ == "__main__":
    main()
