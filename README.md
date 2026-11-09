# chimera

Physically modelled singing and speaking voices for [Hermes](https://github.com/NousResearch/hermes-agent), running
locally on your CPU. chimera is a Hermes plugin that

- reads Hermes' replies aloud as a **text-to-speech provider** (English), and
- lets the agent **sing scores with up to 8 voices** - choirs, canons, duets, chants, spoken lines over a beat - and
  hand you the result as an audio file.

The voices are not samples: a simulated vocal tract (glottis, tongue, lips, nasal tract) is driven by a phonetic
planner, so the same engine speaks and sings at any pitch. It is a Python port of
[chimera-vocal-synth](https://github.com/jamesondh/chimera-vocal-synth) by Jameson Hodge (MIT); where it matches the
original sample for sample and where it deliberately differs is described under
[Fidelity to chimera-vocal-synth](#fidelity-to-chimera-vocal-synth). No account, no API key, no network access.

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
the agent fetches the result with `chimera_job` (or stops it with `cancel: true`). A render belongs to the
conversation that started it, also after Hermes compressed the conversation's context; other conversations cannot
fetch or stop it. When the conversation's session ends (`/new`, leaving the CLI, closing the session in the desktop
app, stopping the gateway) its unfinished renders stop; `/chimera cancel` stops them by hand (Hermes 0.22.6 or
newer; on older Hermes ask the agent to cancel). Finished results are kept for two hours. Results are files under
`<HERMES_HOME>/cache/audio/chimera/`; the agent puts a `MEDIA:` line into its reply and Hermes delivers the file (a
player in the desktop app, an audio file in messaging apps).

Limits per score: 8 voices, 300 s of music, 1200 s summed over the voices; one tempo; one melody line per voice.
A score can reproduce the original's `chimera render notes` output exactly; see
[Fidelity to chimera-vocal-synth](#fidelity-to-chimera-vocal-synth) for the conditions.

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
| `faithful` | `false` | speak like the original's command-line renderer `chimera render text`: no text preparation, no split into phrase windows, the first pitch glides in from 110 Hz, the voice's own random seeds - sample for sample that command's output (each spoken reply then starts with that glide). The original's web app runs the same engine with other defaults (male 110 Hz instead of 112 Hz, child 260 Hz instead of 250 Hz) at the browser's sample rate and volume, so it sounds slightly different |

Changes take effect with the next render; no restart is needed. A Hermes process that serves several profiles
renders each request with the settings of the profile that made it, fixed when the request starts (the worker pool
is shared, `workers` is the last value applied). Environment variables `CHIMERA_BACKEND` and `CHIMERA_WORKERS`
override `backend` and `workers`.
The tool `chimera_speak` also takes `faithful` per call.
`tts.chimera.max_text_length` (Hermes) sets how much text one synthesis call receives (default 4000 characters).

## Disclosure: what chimera does on your computer

- **Subprocesses.** Rendering runs in worker processes started from Hermes' own Python
  (`python -I -S -u -c <bootstrap>`); they exit after 60 s idle and when Hermes exits. numba is only ever imported in
  these workers, so a fault in compiled code cannot take Hermes down. When ffmpeg is on `PATH`, chimera runs it as a
  subprocess to make MP3 files of music (`ffmpeg -hide_banner -loglevel error -y -i <wav> -codec:a libmp3lame -q:a 2
  <mp3>`, no stdin, time limit, no console window); without ffmpeg music stays WAV.
- **Background work.** Long renders continue on a thread in the Hermes process after the tool call returns; results
  are kept two hours. A render stops when its session ends (chimera registers the hooks `on_session_finalize` and
  `on_session_reset`), with `chimera_job` and `cancel: true`, or with `/chimera cancel`.
- **Files written.** Only `<HERMES_HOME>/cache/audio/chimera/` (rendered audio, deleted after `keep_days`) and
  `<HERMES_HOME>/cache/chimera/`: `numba/` (compiled kernels, one folder per chimera version; the folder in use is
  kept, folders of other versions are deleted after `keep_days`) and `tmp/` (temporary files of running renders -
  speech windows, voice tracks, the mix sum - deleted when the render ends, leftovers of an interrupted render after
  a day). Speech is rendered to disk window by window, so Hermes' memory does not grow with the length of the text.
  Folders are created readable by the owner only (0700) and rendered files are 0600, like Hermes' own media cache.
- **Files read outside its own directory.** Hermes' configuration (read-only: `tts`, `timeouts`), Hermes' install
  record for this plugin (`<HERMES_HOME>/plugins/.install-metadata.json`, to show the right install id for
  chimera-numba) and Hermes' session store (`<HERMES_HOME>/state.db`, read-only through Hermes' own `SessionDB`,
  only to tell whether a session continues another one after a context compression). chimera never writes Hermes'
  configuration; `/chimera voice` stores only the plugin's own setting.
- **Network, credentials, telemetry.** None.
- **Security scanner.** `hermes plugins validate` passes (`security scan — safe`). Hermes' install-time scanner
  itself reports only medium-severity notes (verdict `safe`). In the code Hermes runs: `python_subprocess` (the
  render workers in `chimera_vocal/scheduler.py`, ffmpeg in `chimera_hermes/media.py`), `hex_encoded_string` and
  `unicode_escape_chain` (the table of whitespace characters in `chimera_vocal/text/english.py`, written as
  escapes) and `oversized_file` (`chimera_vocal/data/cmudict.txt`, 3.3 MB, see below). Everything else is in the
  test suite and the developer tools: their subprocess calls (`python_subprocess`; `git` and `rustc` in
  `tools/vector_export`), hostile file names (parent-directory paths, the system's password file) that the tests
  check are made harmless (`path_traversal`, `system_passwd_access`), the worker tests' check of the environment
  a worker sees (`dump_all_env`), invisible characters and raw bytes the tests feed in on purpose
  (`invisible_unicode`, `hex_encoded_string`), a lazy import of the `os` module in a test option
  (`python_import_os`) and the instruction to clone the original in `tools/vector_export/README.md` (`git_clone`).
  The CMU pronouncing dictionary is shipped decompressed and byte-identical to the template's data (the template
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

## Fidelity to chimera-vocal-synth

chimera's engine is a port of the engine of chimera-vocal-synth (commit `092b16ce12b9`), checked against the
original's own command-line renderer (`chimera render ...`), which computes the same samples as the web app's engine
for the same input. "Identical" below means: the same length, and no sample differs by more than 4e-8 of the peak
(floating-point rounding, at most 1/750 of the smallest step of a 16-bit file).

**Identical to the original**

- The engine: sustained vowels and spoken text in the male, female and child bodies and in blends of them.
- Scores in which every voice enters at beat 0 (no `start`, no leading rest), `lead_in` stays at its default 0.3 s,
  `normalize` is `false` and every voice has `"seed": 0` (the first voice uses the original's seeds anyway): each
  voice sings what `chimera render notes` sings for the same notes and lyrics, without a single differing 16-bit
  sample (measured on single melodies and a two-voice duet).
- Speech with the `faithful` setting (or `faithful: true` on `chimera_speak`, see Settings): the whole text in one
  piece, starting with the original's glide up from 110 Hz, as `chimera render text` does. (The web app's Speak mode
  uses other default pitches - male 110 Hz, child 260 Hz - and the browser's sample rate and volume.) Measured on
  three new sentences that text preparation leaves unchanged, in the male, female and child voices: same length,
  largest difference 5e-9 of the peak.
- The pronouncing dictionary (decompressed from the original's `cmudict.txt.gz`) and the phoneme tables are the
  original's data byte for byte (sha256 in `THIRD_PARTY_NOTICES.md`).

**Deliberately different**

- *Spoken replies (the default).* Hermes hands over a reply sentence by sentence and does not say which sentence
  comes first. chimera therefore
  - starts every call at the voice's speaking pitch. The original glides into its first voiced sound from 110 Hz:
    female and child voices begin up to 2.6 semitones flat (-215 to -260 cents) and reach their pitch within
    45-70 ms; the male voice (112 Hz) hardly moves. Read sentence by sentence, that glide would open every sentence.
  - renders longer text in pieces split at pauses (commas, sentence ends), so the first audio is ready early and
    the pieces render in parallel. Each piece starts at its settled pitch and with its own random seed; with one
    seed for all pieces, each would repeat the same pitch drift and jitter.

  Length and timing stay exactly those of the original; the sound is a different random take of the same speech,
  not the same samples. Measured on three sentences (1.5 s, 3.6 s, 11.9 s) in the male, female and child voices:
  pitch differs by 4-14 cents (90th percentile over frames voiced in both), the long-term spectrum by 0.07-0.35 dB
  (RMS over 21 third-octave bands, 100 Hz-10 kHz), the loudness envelope by 0.8-1.9 dB (95th percentile). The
  original differs from itself by 8-18 cents, 0.06-0.62 dB and 0.8-2.2 dB when only its random seed changes
  (32 seeds), so the difference is the size of picking another seed. The exception is the missing opening glide: in
  a short one-phrase sentence it raises the pitch difference of the female and child voices to 34 cents (7-9 cents
  without the first 0.1 s).
- *Late entries in a score* (`start` > 0 or a leading rest) are rendered from about 0.3 s before the voice's first
  sound instead of from the beginning of the piece, so no render time goes into the silence before the entry (and
  the per-score voice budget counts only what is rendered). Timing and loudness match the original: the voice's
  first sound starts within 17 samples (0.35 ms) of the original's, each later note within 59 samples before to 102
  samples after it (up to 2.1 ms, less than one period of the note's pitch), the envelope within about 1 dB. Pitch
  drift and vibrato phase are a different random take: 9-11 cents apart without vibrato and 17-36 cents with
  vibrato, less than the original differs from itself between two seeds with vibrato (58-62 cents).
- *Voices after the first* get their own random seed unless they set `"seed": 0`, so voices with the same settings
  do not drift and waver in lockstep.
- *Additions the original does not have:* dynamics (`dyn`), spoken slots (`say`), volume, pan and stereo mixing,
  normalisation to -1 dBFS, text preparation (numbers, abbreviations, accents), and the voices beyond male, female
  and child.

**numba and numpy** compute the same audio up to rounding: at most 5.2e-9 of the peak on the original's reference
inputs, up to 9.9e-9 measured on other input (a voice with a derived seed); the 16-bit files they wrote were
byte-identical in all comparisons made.

## Licence

MIT (`LICENSE`). The engine is a port of chimera-vocal-synth (MIT, Copyright (c) 2026 Jameson Hodge); the English
dictionary is CMUdict (Carnegie Mellon University, BSD-style licence). Full texts and the provenance of the bundled
data: `THIRD_PARTY_NOTICES.md`.
