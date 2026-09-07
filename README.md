# recording-transcriber

One script that turns a meeting recording (Teams/SharePoint link or a local
video/audio file) into a text transcript with speaker labels. Runs fully
locally on your GPU with [WhisperX](https://github.com/m-bain/whisperX) and
[pyannote](https://github.com/pyannote/pyannote-audio). Nothing is uploaded
anywhere.

Input can be a SharePoint share link, any direct URL, or a local video/audio file:

```bash
# Teams recording shared as an "anyone with the link" SharePoint link
uv run transcribe.py run "https://contoso-my.sharepoint.com/:v:/g/personal/user_contoso_com/IQAbc…?e=xyz" -l cs

# a file you already have (mp4, mkv, mp3, wav… anything ffmpeg reads)
uv run transcribe.py run meeting.mp4 -l cs
```

At the end the script shows you who talked the most, with sample sentences, and
asks you to name them:

```
SPEAKER_03  1.6 min (44%), 0:00 – 3:59
   [0:49] So you just pick it there, I'm not going to…
   [3:38] No, it's fixed, unless the study programme has…
   label for SPEAKER_03 [speaker]: teacher
```

Output: `work/<name>/<name>.txt` (paragraph per speaker turn) and
`<name>.srt` (subtitles with speaker prefix).

## Requirements

- Linux/macOS, Python 3.10–3.12, [`uv`](https://docs.astral.sh/uv/) (handles the Python deps)
- `ffmpeg` on PATH
- NVIDIA GPU with ~6 GB free VRAM for `large-v3` (diarization adds ~2 GB); `--device cpu` works but takes hours for long recordings
- ~10 GB disk: ~5 GB Python env with torch, ~3 GB `large-v3`, plus the recording and its WAV
- For speaker labels: a free HuggingFace account
  1. create a **Read** token at <https://huggingface.co/settings/tokens>
  2. click *Agree and access repository* on
     [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and
     [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
     (MIT licensed, the form just asks for name and affiliation)
  3. `python -c 'from huggingface_hub import login; login()'` and paste the token

  The script checks all of this *before* spending 20 minutes on transcription.
  Skip speaker labels entirely with `--no-diarize`.

## Usage

### One shot

```bash
uv run transcribe.py run INPUT [-l cs] [-m large-v3] [-b 4] [--no-diarize] [--default-label speaker] [-y]
```

Runs every stage and asks you to name the speakers at the end. Everything
lands in `work/<name>/`.

### Step by step

Every stage is its own command. Each one is idempotent (skips work that is
already done), prints what it did and what to run next, and reads/writes fixed
file names in the work dir. This is the mode to use from scripts, or when an AI
agent drives the tool: it can run one command, read the output, decide, and
continue.

```bash
uv run transcribe.py status      -w work/x [--json]         # what exists, what is next
uv run transcribe.py download    URL          -w work/x      # SharePoint share link or direct URL -> media.*
uv run transcribe.py audio       [MEDIA]      -w work/x      # ffmpeg -> audio.wav
uv run transcribe.py transcribe  -w work/x -l cs [--force]   # WhisperX -> transcript.json
uv run transcribe.py diarize     -w work/x [--force]         # pyannote -> diarized.json
uv run transcribe.py speakers show -w work/x [--top 8] [--samples 3] [--json]
uv run transcribe.py speakers set  -w work/x SPEAKER_03=teacher SPEAKER_07=teacher --default student
uv run transcribe.py speakers ask  -w work/x                 # interactive alternative to `set`
uv run transcribe.py write       -w work/x [--only teacher] [--name out]   # -> transcript.txt / .srt
```

`speakers show` lists speakers by talk time with their time range and the
longest sentences they said, which is usually enough to tell who is who.
`speakers set` validates the ids, merges into the existing `speakers.json`
(`--clear` to start over) and two ids may share one label. `write --only PREFIX`
keeps only labels starting with PREFIX and marks the omitted stretches with
their time range.

Work dir layout:

```
work/x/
  meta.json         source, duration, model, language
  media.mp4         downloaded recording (only when `download` was used)
  audio.wav         mono 16 kHz
  transcript.json   WhisperX segments with word timings
  diarized.json     same, with "speaker" on every word/segment
  speakers.json     {"default": "student", "speakers": {"SPEAKER_03": "teacher"}}
  transcript.txt    paragraph per speaker turn
  transcript.srt    subtitles with [label] prefix
```

Delete a file to redo that stage, or pass `--force` to `transcribe` / `diarize`.
Redoing `transcribe` removes a stale `diarized.json` automatically.

### For AI agents

Suggested loop: `status --json` → run the command it names → `speakers show
--json` → pick labels from the samples → `speakers set …` → `write`. All
commands exit non-zero with a one-line `ERROR:` on stderr when a precondition
is missing (no token, no access to a gated model, wrong speaker id, missing
stage), so nothing needs a terminal or stdin except `speakers ask` and
`run` without `-y`.

## SharePoint links

Anonymous "anyone with the link" share links (`/:v:/g/personal/...`) are
downloaded directly. Links that require a login are not supported: download the
file in your browser and pass the path instead. Meeting chat is never part of
the recording file, only the Teams conversation has it.

## Notes from real use

Three-hour Czech university info session, RTX 3060 12 GB:

| stage | time |
|---|---|
| WhisperX large-v3, batch 4 | ~10 min |
| pyannote diarization | ~5 min |

`--batch-size 16` ran out of GPU memory after ~55 % of the audio; the script
now halves the batch size and continues instead of failing. Diarization found
45 voices in a Q&A session; naming the top 3 and labelling the rest `student`
was all that was needed.

## Česky

Skript stáhne záznam (SharePoint odkaz nebo soubor), přepíše ho přes WhisperX,
rozpozná mluvčí přes pyannote a na konci se zeptá, kdo je kdo. Výstup je
`.txt` s odstavci podle mluvčího a `.srt` s časy. Vše běží lokálně, potřebuje
NVIDIA GPU, ffmpeg a pro mluvčí zdarma HuggingFace token (postup výše).

## License

MIT
