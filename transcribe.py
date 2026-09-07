#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["whisperx>=3.8", "huggingface_hub>=0.25"]
# ///
"""
transcribe.py – meeting recording -> transcript with speaker labels, fully local.

One-shot (interactive speaker naming at the end):
    uv run transcribe.py run <video|audio|url> -l cs

Step by step (each command is idempotent and prints what it did; good for scripts and AI agents):
    uv run transcribe.py status      -w work/x               what exists, what is next
    uv run transcribe.py download    <url>        -w work/x  SharePoint share link or direct URL
    uv run transcribe.py audio       <media>      -w work/x  ffmpeg -> mono 16 kHz wav
    uv run transcribe.py transcribe  -w work/x -l cs         WhisperX + word alignment
    uv run transcribe.py diarize     -w work/x               pyannote speaker labels (needs HF token)
    uv run transcribe.py speakers show -w work/x [--json]    who talked how much, with sample lines
    uv run transcribe.py speakers set  -w work/x SPEAKER_03=teacher SPEAKER_07=teacher --default student
    uv run transcribe.py speakers ask  -w work/x             interactive naming (terminal)
    uv run transcribe.py write       -w work/x [--only teacher]   txt + srt (optionally only some labels)

Work dir layout (fixed names, so any step can be re-run or done by hand):
    meta.json  media.*  audio.wav  transcript.json  diarized.json  speakers.json  transcript.txt  transcript.srt
Add --json to status / speakers show for machine-readable output.
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
    print(f"» {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def fmt_ts(t: float, sep: str = ",") -> str:
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", sep)


def fmt_min(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


class Work:
    """Fixed file layout inside the work dir."""

    def __init__(self, path: Path):
        self.dir = path
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = path / "meta.json"
        self.wav = path / "audio.wav"
        self.transcript = path / "transcript.json"
        self.diarized = path / "diarized.json"
        self.speakers = path / "speakers.json"
        self.txt = path / "transcript.txt"
        self.srt = path / "transcript.srt"

    @property
    def media(self) -> Path | None:
        found = sorted(p for p in self.dir.glob("media.*") if p.suffix not in (".cookies", ".part"))
        return found[0] if found else None

    def read_meta(self) -> dict:
        return json.load(open(self.meta)) if self.meta.exists() else {}

    def update_meta(self, **kv) -> None:
        m = self.read_meta(); m.update(kv)
        json.dump(m, open(self.meta, "w"), ensure_ascii=False, indent=2)

    def result(self) -> dict:
        """Best available transcript (diarized if present)."""
        for p in (self.diarized, self.transcript):
            if p.exists():
                return json.load(open(p))
        die(f"no transcript in {self.dir} – run `transcribe` first")


# ----------------------------------------------------------------------------- download

SHAREPOINT_RE = re.compile(r"https://([^/]+)/:[a-z]:/g/personal/([^/]+)/([A-Za-z0-9_-]+)")


def resolve_download_url(url: str) -> str:
    m = SHAREPOINT_RE.match(url)
    if m:
        host, user, share_id = m.groups()
        return f"https://{host}/personal/{user}/_layouts/15/download.aspx?share={share_id}"
    return url


def stage_download(w: Work, url: str) -> Path:
    if w.media:
        log(f"download: already have {w.media.name} ({w.media.stat().st_size / 1e6:.0f} MB), skipping")
        return w.media
    direct = resolve_download_url(url)
    ext = Path(urllib.parse.urlparse(direct).path).suffix if not SHAREPOINT_RE.match(url) else ""
    part = w.dir / "media.part"
    log(f"download: {direct}")
    if not shutil.which("curl"):
        die("curl not found")
    cj = w.dir / "media.cookies"
    out = subprocess.run(["curl", "-sS", "-L", "-c", str(cj), "-b", str(cj), "-o", str(part),
                          "-w", "%{http_code} %{content_type} %{filename_effective}", direct],
                         capture_output=True, text=True)
    cj.unlink(missing_ok=True)
    status = out.stdout.strip()
    if out.returncode != 0 or not status.startswith("200"):
        part.unlink(missing_ok=True)
        die(f"download failed ({status or out.stderr.strip()}). If the link needs a login, "
            "download in a browser and use `audio <file>` instead.")
    if "text/html" in status:
        part.unlink(missing_ok=True)
        die("got an HTML page instead of media – the link needs a login. Download in a browser and use `audio <file>`.")
    if not ext:
        ctype = status.split()[1] if len(status.split()) > 1 else ""
        ext = {"video/mp4": ".mp4", "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-m4a": ".m4a",
               "video/webm": ".webm", "video/x-matroska": ".mkv"}.get(ctype.split(";")[0], ".bin")
    dest = w.dir / f"media{ext}"
    part.rename(dest)
    w.update_meta(source=url)
    log(f"download: saved {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


# ----------------------------------------------------------------------------- audio

def stage_audio(w: Work, src: Path | None) -> Path:
    if w.wav.exists():
        log(f"audio: {w.wav.name} exists, skipping")
        return w.wav
    src = src or w.media
    if src is None or not src.exists():
        die("audio: no media file – pass a path or run `download` first")
    if not shutil.which("ffmpeg"):
        die("ffmpeg not found – install it (pacman -S ffmpeg / apt install ffmpeg / brew install ffmpeg)")
    log(f"audio: extracting mono 16 kHz wav from {src.name}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(w.wav)], check=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                "csv=p=0", str(w.wav)], capture_output=True, text=True).stdout or 0)
    w.update_meta(source=w.read_meta().get("source", str(src)), duration_s=round(dur, 1))
    log(f"audio: {w.wav.name}, {dur / 60:.1f} min")
    return w.wav


# ----------------------------------------------------------------------------- preflight

PYANNOTE_MODELS = ["pyannote/speaker-diarization-community-1", "pyannote/segmentation-3.0"]


def check_gpu(device: str) -> None:
    if device != "cuda":
        return
    import torch
    if not torch.cuda.is_available():
        die("CUDA not available – use --device cpu (slow) or fix the GPU setup")
    free, total = torch.cuda.mem_get_info()
    log(f"gpu: {torch.cuda.get_device_name(0)}, {free / 2**30:.1f} GiB free of {total / 2**30:.1f}")


def check_hf() -> None:
    from huggingface_hub import get_token, auth_check
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    if not get_token():
        die("no HuggingFace token. Diarization needs one:\n"
            "  1. create a Read token at https://huggingface.co/settings/tokens\n"
            "  2. run:  python -c 'from huggingface_hub import login; login()'")
    for repo in PYANNOTE_MODELS:
        try:
            auth_check(repo)
        except GatedRepoError:
            die(f"no access to gated model {repo}.\n"
                f"  open https://huggingface.co/{repo}, click 'Agree and access repository', re-run.")
        except RepositoryNotFoundError:
            die(f"model {repo} not found (renamed?) – see https://huggingface.co/pyannote")
    log("huggingface: token OK, pyannote models accessible")


# ----------------------------------------------------------------------------- transcribe

def stage_transcribe(w: Work, model_name: str, language: str | None, device: str, batch_size: int,
                     force: bool = False) -> dict:
    if w.transcript.exists() and not force:
        log(f"transcribe: {w.transcript.name} exists, skipping (use --force to redo)")
        return json.load(open(w.transcript))
    if not w.wav.exists():
        die("transcribe: audio.wav missing – run `audio` first")
    check_gpu(device)
    import torch
    import whisperx
    compute = "float16" if device == "cuda" else "int8"
    log(f"transcribe: loading {model_name} ({compute}) on {device}")
    model = whisperx.load_model(model_name, device, compute_type=compute, language=language)
    audio = whisperx.load_audio(str(w.wav))
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
    json.dump(result, open(w.transcript, "w"), ensure_ascii=False)
    w.update_meta(model=model_name, language=lang)
    if w.diarized.exists():
        w.diarized.unlink()
        log("transcribe: removed stale diarized.json")
    words = sum(len(s["text"].split()) for s in result["segments"])
    log(f"transcribe: {len(result['segments'])} segments, {words} words -> {w.transcript.name}")
    return result


# ----------------------------------------------------------------------------- diarize

def stage_diarize(w: Work, device: str, force: bool = False) -> dict:
    if w.diarized.exists() and not force:
        log(f"diarize: {w.diarized.name} exists, skipping (use --force to redo)")
        return json.load(open(w.diarized))
    if not w.transcript.exists():
        die("diarize: transcript.json missing – run `transcribe` first")
    if not w.wav.exists():
        die("diarize: audio.wav missing – run `audio` first")
    check_gpu(device)
    check_hf()
    import whisperx
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers
    log("diarize: running pyannote (a few minutes)")
    audio = whisperx.load_audio(str(w.wav))
    result = json.load(open(w.transcript))
    segs = DiarizationPipeline(device=device)(audio)
    result = assign_word_speakers(segs, result)
    json.dump(result, open(w.diarized, "w"), ensure_ascii=False)
    n = len({s.get("speaker") for s in result["segments"]})
    log(f"diarize: {n} speakers -> {w.diarized.name}. Next: `speakers show`")
    return result


# ----------------------------------------------------------------------------- speakers

def speaker_stats(result: dict, samples: int = 3) -> list[dict]:
    stats: dict[str, dict] = {}
    for seg in result["segments"]:
        spk = seg.get("speaker", "UNKNOWN")
        st = stats.setdefault(spk, {"id": spk, "seconds": 0.0, "first_s": seg["start"], "last_s": seg["end"],
                                    "segments": 0, "_segs": []})
        st["seconds"] += seg["end"] - seg["start"]
        st["last_s"] = seg["end"]
        st["segments"] += 1
        st["_segs"].append(seg)
    out = []
    for st in sorted(stats.values(), key=lambda s: -s["seconds"]):
        longest = sorted(st.pop("_segs"), key=lambda x: -(x["end"] - x["start"]))[:samples]
        st["samples"] = [{"at": fmt_min(x["start"]), "text": x["text"].strip()}
                         for x in sorted(longest, key=lambda x: x["start"])]
        st["seconds"] = round(st["seconds"], 1)
        out.append(st)
    return out


def load_speakers_cfg(w: Work) -> dict:
    return json.load(open(w.speakers)) if w.speakers.exists() else {"default": None, "speakers": {}}


def cmd_speakers_show(w: Work, top: int, as_json: bool, samples: int) -> None:
    result = w.result()
    if not any("speaker" in s for s in result["segments"]):
        die("no speaker labels yet – run `diarize` first")
    stats = speaker_stats(result, samples)
    cfg = load_speakers_cfg(w)
    total = sum(s["seconds"] for s in stats) or 1
    for s in stats:
        s["share"] = round(100 * s["seconds"] / total, 1)
        s["label"] = cfg["speakers"].get(s["id"], cfg.get("default"))
    if as_json:
        print(json.dumps({"speakers": stats[:top], "total_speakers": len(stats), "default": cfg.get("default")},
                         ensure_ascii=False, indent=2))
        return
    print(f"{len(stats)} speakers, showing top {min(top, len(stats))} by talk time. "
          f"default label: {cfg.get('default') or '(none, keeps SPEAKER_XX)'}\n")
    for s in stats[:top]:
        lb = f" -> {s['label']}" if s["label"] else ""
        print(f"{s['id']}{lb}  {s['seconds'] / 60:.1f} min ({s['share']:.0f}%), "
              f"{fmt_min(s['first_s'])}–{fmt_min(s['last_s'])}, {s['segments']} segments")
        for smp in s["samples"]:
            t = smp["text"]
            print(f"   [{smp['at']}] {t[:160]}{'…' if len(t) > 160 else ''}")
        print()
    rest = stats[top:]
    if rest:
        print(f"... {len(rest)} more speakers, {sum(s['seconds'] for s in rest) / 60:.1f} min total")
    print("\nNext: `speakers set SPEAKER_XX=label ... --default label`, then `write`.")


def cmd_speakers_set(w: Work, pairs: list[str], default: str | None, clear: bool) -> None:
    cfg = {"default": None, "speakers": {}} if clear else load_speakers_cfg(w)
    known = {s.get("speaker") for s in w.result()["segments"]}
    for p in pairs:
        if "=" not in p:
            die(f"expected SPEAKER_XX=label, got {p!r}")
        k, v = p.split("=", 1)
        if k not in known:
            die(f"{k} is not a speaker in this transcript (see `speakers show`)")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", v):
            die(f"label {v!r}: use letters, digits, _ . - only")
        cfg["speakers"][k] = v
    if default is not None:
        cfg["default"] = default
    json.dump(cfg, open(w.speakers, "w"), ensure_ascii=False, indent=2)
    log(f"speakers: saved {w.speakers} ({len(cfg['speakers'])} named, default={cfg['default']!r})")


def cmd_speakers_ask(w: Work, top: int, default_label: str) -> None:
    result = w.result()
    stats = speaker_stats(result)
    total = sum(s["seconds"] for s in stats) or 1
    print("=" * 72)
    print(f" {len(stats)} speakers. Name the top {min(top, len(stats))}; Enter = default, '=' = same as previous.")
    print("=" * 72)
    mapping: dict[str, str] = {}
    prev = None
    for s in stats[:top]:
        print(f"\n{s['id']}  {s['seconds'] / 60:.1f} min ({100 * s['seconds'] / total:.0f}%), "
              f"{fmt_min(s['first_s'])}–{fmt_min(s['last_s'])}")
        for smp in s["samples"]:
            print(f"   [{smp['at']}] {smp['text'][:160]}")
        while True:
            ans = input(f"   label for {s['id']} [{default_label}]: ").strip()
            if ans == "=" and prev:
                ans = prev
            if ans and not re.fullmatch(r"[A-Za-z0-9_.-]+", ans):
                print("   use letters, digits, _ . - only")
                continue
            break
        if ans:
            mapping[s["id"]] = ans; prev = ans
    rest = len(stats) - min(top, len(stats))
    default = default_label
    if rest > 0:
        default = input(f"\nlabel for the remaining {rest} speakers [{default_label}]: ").strip() or default_label
    json.dump({"default": default, "speakers": mapping}, open(w.speakers, "w"), ensure_ascii=False, indent=2)
    log(f"speakers: saved {w.speakers}")


# ----------------------------------------------------------------------------- write

def stage_write(w: Work, only: str | None = None, out_stem: str | None = None) -> tuple[Path, Path]:
    result = w.result()
    cfg = load_speakers_cfg(w)
    has_speakers = any("speaker" in s for s in result["segments"])
    if only and not has_speakers:
        die("--only needs speaker labels – run `diarize` first")

    def label(seg):
        if not has_speakers:
            return None
        spk = seg.get("speaker", "UNKNOWN")
        return cfg["speakers"].get(spk, cfg.get("default") or spk)

    stem = out_stem or ("transcript" if not only else f"transcript-{only}")
    txt, srt = w.dir / f"{stem}.txt", w.dir / f"{stem}.srt"
    kept = skipped = words = 0
    with open(txt, "w") as f, open(srt, "w") as g:
        last = object(); gap = None; n = 0
        for seg in result["segments"]:
            text = seg["text"].strip(); lb = label(seg)
            if only and not (lb or "").startswith(only):
                skipped += 1
                if gap is None: gap = seg["start"]
                continue
            kept += 1; words += len(text.split())
            if gap is not None:
                f.write(f"\n\n[… {fmt_ts(gap, '.')[:-4]}–{fmt_ts(seg['start'], '.')[:-4]} omitted]")
                gap = None; last = object()
            if lb is None:
                f.write(text + "\n")
            else:
                if lb != last:
                    f.write(f"\n\n[{lb}] "); last = lb
                f.write(text + " ")
            n += 1
            pre = f"[{lb}] " if lb else ""
            g.write(f"{n}\n{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}\n{pre}{text}\n\n")
    extra = f", kept {kept} / skipped {skipped} segments" if only else ""
    log(f"write: {txt} ({words} words{extra})")
    log(f"write: {srt}")
    return txt, srt


# ----------------------------------------------------------------------------- status

def cmd_status(w: Work, as_json: bool) -> None:
    meta = w.read_meta()
    st = {
        "workdir": str(w.dir),
        "media": w.media.name if w.media else None,
        "audio": w.wav.exists(),
        "transcript": w.transcript.exists(),
        "diarized": w.diarized.exists(),
        "speakers_named": w.speakers.exists(),
        "written": w.txt.exists(),
        "meta": meta,
    }
    if not st["audio"] and not st["media"]:
        nxt = "download <url>  or  audio <media-file>"
    elif not st["audio"]:
        nxt = "audio"
    elif not st["transcript"]:
        nxt = "transcribe"
    elif not st["diarized"]:
        nxt = "diarize  (or `write` for a transcript without speakers)"
    elif not st["speakers_named"]:
        nxt = "speakers show, then speakers set ... --default ..."
    else:
        nxt = "write  (done if transcript.txt is current)"
    st["next"] = nxt
    if as_json:
        print(json.dumps(st, ensure_ascii=False, indent=2))
        return
    for k in ("media", "audio", "transcript", "diarized", "speakers_named", "written"):
        v = st[k]
        mark = "✓" if v else "·"
        print(f"  {mark} {k}" + (f": {v}" if isinstance(v, str) else ""))
    if meta:
        print("  meta: " + ", ".join(f"{k}={v}" for k, v in meta.items()))
    print(f"\nnext: {nxt}")


# ----------------------------------------------------------------------------- run (one-shot)

def cmd_run(w: Work, inp: str, a) -> None:
    is_url = inp.startswith(("http://", "https://"))
    if is_url:
        stage_download(w, inp)
        stage_audio(w, None)
    else:
        p = Path(inp)
        if not p.exists():
            die(f"input not found: {p}")
        stage_audio(w, p)
    if not a.no_diarize:
        check_hf()  # fail early, before the long transcription
    stage_transcribe(w, a.model, a.language, a.device, a.batch_size)
    if not a.no_diarize:
        stage_diarize(w, a.device)
        if w.speakers.exists():
            log(f"speakers: using existing {w.speakers.name}")
        elif sys.stdin.isatty() and not a.yes:
            cmd_speakers_ask(w, a.top, a.default_label)
        else:
            log("speakers: non-interactive, keeping SPEAKER_XX (use `speakers show/set` later)")
    stage_write(w)


# ----------------------------------------------------------------------------- cli

def default_workdir(inp: str | None) -> Path:
    if not inp:
        die("-w/--workdir is required for this command")
    if inp.startswith(("http://", "https://")):
        stem = Path(urllib.parse.urlparse(inp).path).stem if not SHAREPOINT_RE.match(inp) else "recording"
    else:
        stem = Path(inp).stem
    return Path("work") / stem


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_work(p, required=False):
        p.add_argument("-w", "--workdir", required=required, help="work dir (default: ./work/<input stem>)")

    def add_gpu(p):
        p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    p = sub.add_parser("run", help="all stages in one go (interactive speaker naming)")
    p.add_argument("input"); add_work(p); add_gpu(p)
    p.add_argument("-m", "--model", default="large-v3")
    p.add_argument("-l", "--language", default=None)
    p.add_argument("-b", "--batch-size", type=int, default=4)
    p.add_argument("--no-diarize", action="store_true")
    p.add_argument("--default-label", default="speaker")
    p.add_argument("--top", type=int, default=6)
    p.add_argument("-y", "--yes", action="store_true", help="never prompt")

    p = sub.add_parser("status", help="what exists in the work dir and what to do next")
    add_work(p, True); p.add_argument("--json", action="store_true")

    p = sub.add_parser("download", help="fetch media from a URL into the work dir")
    p.add_argument("url"); add_work(p)

    p = sub.add_parser("audio", help="extract audio.wav from a media file (or the downloaded one)")
    p.add_argument("media", nargs="?"); add_work(p)

    p = sub.add_parser("transcribe", help="WhisperX transcription + alignment")
    add_work(p, True); add_gpu(p)
    p.add_argument("-m", "--model", default="large-v3")
    p.add_argument("-l", "--language", default=None, help="e.g. cs, en; default auto-detect")
    p.add_argument("-b", "--batch-size", type=int, default=4)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("diarize", help="pyannote speaker labels")
    add_work(p, True); add_gpu(p); p.add_argument("--force", action="store_true")

    p = sub.add_parser("speakers", help="inspect / name speakers")
    ssub = p.add_subparsers(dest="scmd", required=True)
    q = ssub.add_parser("show"); add_work(q, True)
    q.add_argument("--top", type=int, default=8); q.add_argument("--samples", type=int, default=3)
    q.add_argument("--json", action="store_true")
    q = ssub.add_parser("set"); add_work(q, True)
    q.add_argument("pairs", nargs="*", metavar="SPEAKER_XX=label")
    q.add_argument("--default", help="label for all un-named speakers")
    q.add_argument("--clear", action="store_true", help="start from an empty mapping")
    q = ssub.add_parser("ask"); add_work(q, True)
    q.add_argument("--top", type=int, default=6); q.add_argument("--default-label", default="speaker")

    p = sub.add_parser("write", help="produce transcript.txt / .srt")
    add_work(p, True)
    p.add_argument("--only", metavar="PREFIX", help="keep only speakers whose label starts with PREFIX")
    p.add_argument("--name", help="output file stem (default transcript / transcript-PREFIX)")

    a = ap.parse_args(argv)

    if a.cmd == "run":
        cmd_run(Work(Path(a.workdir) if a.workdir else default_workdir(a.input)), a.input, a)
    elif a.cmd == "status":
        cmd_status(Work(Path(a.workdir)), a.json)
    elif a.cmd == "download":
        w = Work(Path(a.workdir) if a.workdir else default_workdir(a.url))
        stage_download(w, a.url); log(f"next: audio -w {w.dir}")
    elif a.cmd == "audio":
        w = Work(Path(a.workdir) if a.workdir else default_workdir(a.media))
        stage_audio(w, Path(a.media) if a.media else None); log(f"next: transcribe -w {w.dir}")
    elif a.cmd == "transcribe":
        stage_transcribe(Work(Path(a.workdir)), a.model, a.language, a.device, a.batch_size, a.force)
        log("next: diarize (speaker labels) or write (plain transcript)")
    elif a.cmd == "diarize":
        stage_diarize(Work(Path(a.workdir)), a.device, a.force)
    elif a.cmd == "speakers":
        w = Work(Path(a.workdir))
        if a.scmd == "show":
            cmd_speakers_show(w, a.top, a.json, a.samples)
        elif a.scmd == "set":
            cmd_speakers_set(w, a.pairs, a.default, a.clear); log("next: write")
        else:
            cmd_speakers_ask(w, a.top, a.default_label); log("next: write")
    elif a.cmd == "write":
        stage_write(Work(Path(a.workdir)), a.only, a.name)


if __name__ == "__main__":
    main()
