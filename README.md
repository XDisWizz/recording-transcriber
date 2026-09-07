# recording-transcriber

One script that turns a meeting recording (Teams/SharePoint link or a local
video/audio file) into a text transcript with speaker labels. Runs fully
locally on your GPU with [WhisperX](https://github.com/m-bain/whisperX) and
[pyannote](https://github.com/pyannote/pyannote-audio). Nothing is uploaded
anywhere.

```bash
uv run transcribe.py "https://contoso-my.sharepoint.com/:v:/g/personal/user_contoso_com/IQAbc…?e=xyz" -l cs
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

```
uv run transcribe.py INPUT [options]

INPUT                  video/audio file, SharePoint share link, or direct URL
-l, --language cs      language code; default auto-detect
-m, --model large-v3   any faster-whisper model (medium, small… for weaker GPUs)
-b, --batch-size 4     halved automatically on GPU out-of-memory
--no-diarize           transcript only, no speaker labels
--default-label NAME   label for speakers you don't name (default: speaker)
--top N                how many speakers to ask about (default 6)
--rename               only redo the naming step over existing results
-y, --yes              non-interactive; keeps SPEAKER_XX (name later with --rename)
-w, --workdir DIR      intermediate files (default ./work/<name>)
```

Every stage is cached in the work dir, so re-running after a crash or with
`--rename` is instant. Delete the work dir to start over.

Speaker mapping lives in `work/<name>/speakers.json` and can be edited by hand:

```json
{ "default": "student", "speakers": { "SPEAKER_03": "teacher", "SPEAKER_07": "teacher" } }
```

Two labels pointing to the same name are merged, which is handy when
diarization splits one person into two (it happens with a live voice vs. a
pre-recorded clip of the same speaker). Type `=` at the prompt to reuse the
previous label.

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
