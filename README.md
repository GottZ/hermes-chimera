# chimera

Physically modelled singing and speaking voices for [Hermes](https://github.com/NousResearch/hermes-agent), running
locally on your CPU. chimera is a Hermes plugin that

- reads Hermes' replies aloud as a **text-to-speech provider** (English), and
- lets the agent **sing scores with up to 8 voices** - choirs, canons, duets, chants, spoken lines over a beat - and
  hand you the result as an audio file.

The voices are not samples: a simulated vocal tract (glottis, tongue, lips, nasal tract) is driven by a phonetic
planner, so the same engine speaks and sings at any pitch. It is a Python port of
[chimera-vocal-synth](https://github.com/jamesondh/chimera-vocal-synth) by Jameson Hodge (MIT). No account, no API
key, no network access.

Requires Hermes 0.21.6 or newer on Linux, macOS or Windows. Python dependency: `numpy` (installed by Hermes).

## Install

Command line:

    hermes plugins install GottZ/hermes-chimera --yes-deps --enable

`--yes-deps` agrees to installing numpy into Hermes' managed environment (without it Hermes asks).
Desktop app: Capabilities -> Plugins -> Install from repository, enter `GottZ/hermes-chimera`.

## Restart Hermes after installing

A running Hermes process does not see packages installed after it started. Restart Hermes (gateway, desktop app or
CLI session) after installing chimera or its accelerator. `/chimera status` tells you when a restart is still due.

## Use chimera for spoken replies (text-to-speech)

- Desktop app: **Capabilities -> Tools -> Text-to-Speech -> Chimera (local vocal synth)**. (Settings -> Voice then
  shows "chimera" but cannot select it.)
- Command line: `hermes config set tts.provider chimera`, or pick it in `hermes tools` -> Text-to-Speech.

chimera never changes this setting itself. Spoken replies stream as 48 kHz PCM where Hermes supports streaming
(desktop, CLI, TUI); messaging platforms get voice bubbles when ffmpeg is installed (Hermes converts the WAV).

## Choose a voice

- Desktop app: Settings -> Plugins -> chimera -> `voice`.
- In a chat: `/chimera voice female` (`/chimera voices` lists them).
- Command line: `hermes config set tts.voice female` (Hermes 0.22.6+; older versions need `--force`).

The plugin setting wins over `tts.voice`; a voice of another provider (for example `alloy`) is ignored with a log line.

| voice | |
|---|---|
| `male` | the template's default speaker |
| `female` | |
| `child` | |
| `androgynous` | half male, half female body |
| `teen` | half female, half child body, slightly faster |
| `male-deep` | lower, darker |
| `female-bright` | higher, brighter |
| `narrator` | calm, slower male |
| `whisper` | unvoiced female |
| `male-whisper` | unvoiced male |
| `robot` | monotone |
| `elder` | rough, breathy male |

A voice can be adjusted with options: `female;pitch=230;rate=1.1`, `male;body=0.3,0` (female/child share of the
body), `accent=`, `breathiness=`, `tension=`, `brightness=`, `roughness=`, `whisper=true`, `seed=`.

## Faster rendering with chimera-numba

chimera works out of the box with numpy. The optional companion plugin `chimera-numba` installs numba (about 210 MB
with llvmlite), which makes rendering about 19x faster:

    hermes plugins install 'GottZ/hermes-chimera#accel' --yes-deps --enable

Keep the quotes (zsh with `extendedglob` treats `#` specially); `GottZ/hermes-chimera/accel` is the same id. In the
desktop app use the same id in Capabilities -> Plugins -> Install from repository. Restart Hermes, then run
`/chimera warmup` (or `hermes chimera warmup`) once: the kernels are compiled into a cache and checked against a
reference before they are used. If compilation or the check fails, chimera stays on numpy and `/chimera status`
says why; `/chimera warmup` tries again. If the compiled cache goes missing later, the next render compiles it again.

numba has no wheels for macOS x86_64, musl Linux (Alpine) and Android; there chimera keeps using numpy.
Removing chimera-numba leaves numba in Hermes' Python environment until Hermes rebuilds it, so chimera keeps using
it; set the plugin setting `backend` to `numpy` to stop.
Installed from the repository as above, the companion has no `.git` folder, so `hermes plugins update` skips it;
update it with `hermes plugins install 'GottZ/hermes-chimera#accel' --force --yes-deps --enable`.

## Music

The agent writes a score (tempo, voices with body, pan and volume, notes with one lyric syllable each, rests, spoken
slots, dynamics) and calls `chimera_render`; the bundled skill `chimera:music` teaches it the format with three
worked examples. Tools:

| tool | |
|---|---|
| `chimera_render` | render a score to audio (WAV, or MP3 when ffmpeg is available) |
| `chimera_check` | validate a score: errors with fixes, syllable alignment, length, estimated render time |
| `chimera_speak` | speak text (up to 4000 characters per call) in a chosen chimera voice into a file |
| `chimera_job` | wait for a render that is still running, or cancel it |

Hermes gives one tool call a fixed time limit (420 s by default). Longer renders keep running in the background and
the agent fetches the result with `chimera_job` (or stops it with `cancel: true`); a job can only be fetched from
the session that started it (or, on Hermes versions that pass a conversation id, from the same conversation after
the session id changed), and results are kept for two hours. Results are files under `<HERMES_HOME>/cache/audio/chimera/`; the agent puts a `MEDIA:` line into its reply
and Hermes delivers the file (a player in the desktop app, an audio file in messaging apps).

Limits per score: 8 voices, 300 s of music, 1200 s summed over the voices; one tempo; one melody line per voice.

## Performance

Real-time factor (render time ÷ audio length) per voice, measured on an x86_64 Linux desktop CPU (Xeon E-2176G,
6 cores at 3.7 GHz, Python 3.14); slower CPUs take proportionally longer:

| backend | real-time factor (1 voice / 4 at once) | 30 s of speech (4 workers) | 60 s score, 4 voices | first use |
|---|---|---|---|---|
| numba (chimera-numba) | about 0.45 / 0.55 | about 4 s | about 50 s | one-time warm-up about 20 s |
| numpy | about 8.5 / 10 | about 1½ min | about 15 min | about 10 s for the first sentence |

Rendering runs in worker processes (default `min(4, max(2, CPUs-1))`, one kept free for speech so music never delays
spoken replies). Memory: about 50-60 MB per numpy worker, about 160 MB per numba worker (about 325 MB while the
warm-up compiles).

## Settings

Settings -> Plugins -> chimera (stored under `plugins.entries.chimera.settings`):

| key | default | |
|---|---|---|
| `voice` | empty | voice for spoken replies (empty = `tts.voice`, else `male`) |
| `backend` | `auto` | `auto` (numba when installed and warmed up), `numba`, `numpy` |
| `workers` | `0` | render worker processes (0 = automatic) |
| `text_prep` | `true` | expand numbers, abbreviations and accents before reading |
| `music_format` | `auto` | `auto` (MP3 when ffmpeg exists), `wav`, `mp3` |
| `keep_days` | `7` | delete rendered chimera audio older than this (0 = keep) |
| `max_voices` | `8` | largest ensemble one score may have |
| `faithful` | `false` | speak exactly like the original web app's `render text`: no text preparation, no split into phrase windows, the first pitch glides in from 110 Hz, the voice's own random seeds - sample for sample the template's output (each spoken reply then starts with that glide) |

Changes take effect with the next render; no restart is needed. A Hermes process that serves several profiles
renders with the settings of the profile that made the request (the worker pool is shared, `workers` is the last
value applied). Environment variables `CHIMERA_BACKEND` and `CHIMERA_WORKERS` override `backend` and `workers`.
The tool `chimera_speak` also takes `faithful` per call.
`tts.chimera.max_text_length` (Hermes) sets how much text one synthesis call receives (default 4000 characters).

## Disclosure: what chimera does on your computer

- **Subprocesses.** Rendering runs in worker processes started from Hermes' own Python
  (`python -I -S -u -c <bootstrap>`); they exit after 60 s idle and when Hermes exits. numba is only ever imported in
  these workers, so a fault in compiled code cannot take Hermes down. When ffmpeg is on `PATH`, chimera runs it as a
  subprocess to make MP3 files of music (`ffmpeg -hide_banner -loglevel error -y -i <wav> -codec:a libmp3lame -q:a 2
  <mp3>`, no stdin, time limit, no console window); without ffmpeg music stays WAV.
- **Background work.** Long renders continue on a thread in the Hermes process after the tool call returns; results
  are kept two hours; `chimera_job` with `cancel: true` stops one.
- **Files written.** Only `<HERMES_HOME>/cache/audio/chimera/` (rendered audio, deleted after `keep_days`) and
  `<HERMES_HOME>/cache/chimera/`: `numba/` (compiled kernels, one folder per chimera version; the folder in use is
  kept, folders of other versions are deleted after `keep_days`) and `tmp/` (temporary files of running renders -
  speech windows, voice tracks, the mix sum - deleted when the render ends, leftovers of an interrupted render after
  a day). Speech is rendered to disk window by window, so Hermes' memory does not grow with the length of the text.
  Folders are created readable by the owner only (0700) and rendered files are 0600, like Hermes' own media cache.
- **Files read outside its own directory.** Hermes' configuration (read-only: `tts`, `timeouts`) and Hermes' install
  record for this plugin (`<HERMES_HOME>/plugins/.install-metadata.json`, to show the right install id for
  chimera-numba). chimera never writes Hermes' configuration; `/chimera voice` stores only the plugin's own setting.
- **Network, credentials, telemetry.** None.
- **Security scanner.** `hermes plugins validate` passes (`security scan — safe`). Hermes' install-time scanner
  itself lists one medium finding, `oversized_file`, for `chimera_vocal/data/cmudict.txt` (3.3 MB, verdict `safe`):
  the CMU pronouncing dictionary is shipped decompressed and byte-identical to the template's data (the template
  ships it as `cmudict.txt.gz`; the sha256 of the decompressed file is in `THIRD_PARTY_NOTICES.md`), so it can be
  read and checked without unpacking. The catalog would also accept the compressed file.

## Limits

- English only. Other languages are read with English pronunciation rules (accents are removed first).
- With `plugins.isolation: host` Hermes cannot stream from a plugin; spoken replies then arrive sentence by sentence,
  and voice bubbles only where Hermes asks for an Ogg file.
- One tempo per score; one melody line per voice; up to 8 voices and 300 s per score.
- Without the accelerator long scores take minutes (see Performance).
- An unrelated plugin, `jnorthrup/hermes-plugin-chimera`, also installs as `chimera`; Hermes can only have one of the
  two.

## Licence

MIT (`LICENSE`). The engine is a port of chimera-vocal-synth (MIT, Copyright (c) 2026 Jameson Hodge); the English
dictionary is CMUdict (Carnegie Mellon University, BSD-style licence). Full texts and the provenance of the bundled
data: `THIRD_PARTY_NOTICES.md`.
