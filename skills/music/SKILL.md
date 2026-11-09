---
name: music
description: "Write scores for chimera_render: voices, ranges, lyrics per note, canon/choir examples, MEDIA delivery."
version: 0.1.0
metadata:
  hermes:
    tags: [music, singing, audio]
    requires_tools: [chimera_render]
---

# Singing with chimera

chimera sings and speaks with physically modelled human voices (male, female, child and blends) on this computer.
Use it for choirs, canons, duets, chants, sung greetings and spoken lines over a beat. It does not play instruments.

## Workflow

1. Write a score (format below).
2. Call `chimera_check` first when the score has more than 2 voices or lyrics with multi-syllable words. Fix every
   `error` issue; `warning` issues (range, render time) may stay.
3. Call `chimera_render` with the score. Optional `filename` (letters, digits, hyphens).
4. The result has a `media` value like `MEDIA:/path/to/file.mp3`. **Copy that line exactly, on its own line, into
   your reply, outside any code block or quote.** That line is what delivers the audio; without it the user gets
   nothing. Do not add `[[audio_as_voice]]` for music.
5. If the result says `"status": "running"`, call `chimera_job` with the `job_id` (repeat until `done`). Long scores
   on a computer without the numba accelerator take a while; tell the user it is rendering.

## Score format

```
{ "title": "...", "bpm": 84, "voices": [ { "name": "...", "body": "female", "notes": [ ... ] } ] }
```

Score: `bpm` 30-240 (one tempo for the whole score), `voices` 1-8. Optional: `title`, `lead_in` (silence before
beat 0, default 0.3 s, raised automatically when needed - you never have to compute it), `normalize` (default true),
`stereo` ("auto" = stereo when any `pan` is not 0), `format` ("auto" = mp3 when ffmpeg exists, "wav", "mp3").

Voice: `notes` (required), `body` ("male", "female", "child" or a blend like "female=0.5,child=0.2"), `name`,
`start` (beat where this voice enters - use it for canon entries), `volume` (dB, -40..12), `pan` (-1 left .. 1 right),
`breath` (3-20), `tension`, `breathiness`, `brightness`, `roughness`, `ring` (0-1), `vibrato` (cents, 0-100),
`vibrato_rate` (Hz), `seed`, and `lyrics` (whole text, one syllable per sounding note, instead of per-note `lyric`).

Items in `notes`, played back to back from `start`:
- note `{"pitch": "F#4", "beats": 1, "lyric": "row,"}`; `pitch` is a note name (C, C#, Db ... B plus octave;
  C4 = middle C; cents allowed: "A4+15") or a MIDI number 28-100;
- rest `{"rest": 1}` (also ends a phrase: the voice breathes);
- spoken words `{"say": "and now, all together!", "beats": 4}` with optional `rate` (0.5-1.8) and `hz`;
  the slot must be long enough for the words, `chimera_check` says how many beats they need.

Lyrics, one syllable per sounding note:
- `"twin-"` the word continues on the next note (`"twin-", "kle"`);
- `"-"` holds the previous vowel on a new pitch (melisma);
- end a syllable with `,` `.` `;` `:` `!` `?` to end the phrase (a breath follows);
- no lyrics at all: the voice sings "ah". Digits are not sung; write numbers as words. English only.

Dynamics: `"dyn": "pp" | "p" | "mp" | "mf" | "f" | "ff"` on a note applies from that note on. pp/p are clearly softer;
f/ff sound brighter and more pressed - the loudness change mostly comes from the mix.

## Ranges

| part | body | comfortable range | notes |
|---|---|---|---|
| bass / baritone | male | D2-E4 | low notes are quiet; keep the bass above F2 for clarity |
| tenor | male | C3-A4 | |
| alto | female | F3-D5 | |
| soprano | female | C4-C6 | |
| child | child | A3-A5 | |

Outside these ranges the voice sounds strained (warning only); MIDI 28-100 is the hard limit.

## Example 1: four-part cadence (SATB)

```json
{"title":"Sing we now","bpm":72,"voices":[
{"name":"soprano","body":"female","pan":-0.3,"notes":[{"pitch":"E4","beats":1,"lyric":"Sing","dyn":"p"},{"pitch":"F4","beats":1,"lyric":"we"},{"pitch":"G4","beats":1,"lyric":"now,"},{"pitch":"A4","beats":1,"lyric":"sing"},{"pitch":"F4","beats":1,"lyric":"we"},{"pitch":"G4","beats":1,"lyric":"all."},{"pitch":"A4","beats":1,"lyric":"A-","dyn":"f"},{"pitch":"G4","beats":2,"lyric":"men."}]},
{"name":"alto","body":"female","pan":-0.1,"notes":[{"pitch":"C4","beats":1,"lyric":"Sing","dyn":"p"},{"pitch":"C4","beats":1,"lyric":"we"},{"pitch":"E4","beats":1,"lyric":"now,"},{"pitch":"E4","beats":1,"lyric":"sing"},{"pitch":"D4","beats":1,"lyric":"we"},{"pitch":"D4","beats":1,"lyric":"all."},{"pitch":"F4","beats":1,"lyric":"A-","dyn":"f"},{"pitch":"E4","beats":2,"lyric":"men."}]},
{"name":"tenor","body":"male","pan":0.1,"notes":[{"pitch":"G3","beats":1,"lyric":"Sing","dyn":"p"},{"pitch":"A3","beats":1,"lyric":"we"},{"pitch":"C4","beats":1,"lyric":"now,"},{"pitch":"C4","beats":1,"lyric":"sing"},{"pitch":"A3","beats":1,"lyric":"we"},{"pitch":"B3","beats":1,"lyric":"all."},{"pitch":"C4","beats":1,"lyric":"A-","dyn":"f"},{"pitch":"C4","beats":2,"lyric":"men."}]},
{"name":"bass","body":"male","pan":0.3,"notes":[{"pitch":"C3","beats":1,"lyric":"Sing","dyn":"p"},{"pitch":"F3","beats":1,"lyric":"we"},{"pitch":"C3","beats":1,"lyric":"now,"},{"pitch":"A2","beats":1,"lyric":"sing"},{"pitch":"D3","beats":1,"lyric":"we"},{"pitch":"G2","beats":1,"lyric":"all."},{"pitch":"F2","beats":1,"lyric":"A-","dyn":"f"},{"pitch":"C3","beats":2,"lyric":"men."}]}]}
```

## Example 2: canon "Row, Row, Row Your Boat" (traditional) for three voices

Same melody, entries every 8 beats with `start`, spread with `pan`; the male voice sings an octave lower.

```json
{"title":"Row your boat","bpm":100,"voices":[
{"name":"one","body":"female","start":0,"pan":-0.5,"lyrics":"Row, row, row your boat, gent-ly down the stream. Mer-ri-ly, mer-ri-ly, mer-ri-ly, mer-ri-ly, life is but a dream.","notes":[{"pitch":"C4","beats":1},{"pitch":"C4","beats":1},{"pitch":"C4","beats":0.75},{"pitch":"D4","beats":0.25},{"pitch":"E4","beats":1},{"pitch":"E4","beats":0.75},{"pitch":"D4","beats":0.25},{"pitch":"E4","beats":0.75},{"pitch":"F4","beats":0.25},{"pitch":"G4","beats":2},{"pitch":"C5","beats":0.5},{"pitch":"C5","beats":0.25},{"pitch":"C5","beats":0.25},{"pitch":"G4","beats":0.5},{"pitch":"G4","beats":0.25},{"pitch":"G4","beats":0.25},{"pitch":"E4","beats":0.5},{"pitch":"E4","beats":0.25},{"pitch":"E4","beats":0.25},{"pitch":"C4","beats":0.5},{"pitch":"C4","beats":0.25},{"pitch":"C4","beats":0.25},{"pitch":"G4","beats":0.75},{"pitch":"F4","beats":0.25},{"pitch":"E4","beats":0.75},{"pitch":"D4","beats":0.25},{"pitch":"C4","beats":2}]},
{"name":"two","body":"male","start":8,"pan":0,"lyrics":"Row, row, row your boat, gent-ly down the stream. Mer-ri-ly, mer-ri-ly, mer-ri-ly, mer-ri-ly, life is but a dream.","notes":[{"pitch":"C3","beats":1},{"pitch":"C3","beats":1},{"pitch":"C3","beats":0.75},{"pitch":"D3","beats":0.25},{"pitch":"E3","beats":1},{"pitch":"E3","beats":0.75},{"pitch":"D3","beats":0.25},{"pitch":"E3","beats":0.75},{"pitch":"F3","beats":0.25},{"pitch":"G3","beats":2},{"pitch":"C4","beats":0.5},{"pitch":"C4","beats":0.25},{"pitch":"C4","beats":0.25},{"pitch":"G3","beats":0.5},{"pitch":"G3","beats":0.25},{"pitch":"G3","beats":0.25},{"pitch":"E3","beats":0.5},{"pitch":"E3","beats":0.25},{"pitch":"E3","beats":0.25},{"pitch":"C3","beats":0.5},{"pitch":"C3","beats":0.25},{"pitch":"C3","beats":0.25},{"pitch":"G3","beats":0.75},{"pitch":"F3","beats":0.25},{"pitch":"E3","beats":0.75},{"pitch":"D3","beats":0.25},{"pitch":"C3","beats":2}]},
{"name":"three","body":"child","start":16,"pan":0.5,"lyrics":"Row, row, row your boat, gent-ly down the stream. Mer-ri-ly, mer-ri-ly, mer-ri-ly, mer-ri-ly, life is but a dream.","notes":[{"pitch":"C4","beats":1},{"pitch":"C4","beats":1},{"pitch":"C4","beats":0.75},{"pitch":"D4","beats":0.25},{"pitch":"E4","beats":1},{"pitch":"E4","beats":0.75},{"pitch":"D4","beats":0.25},{"pitch":"E4","beats":0.75},{"pitch":"F4","beats":0.25},{"pitch":"G4","beats":2},{"pitch":"C5","beats":0.5},{"pitch":"C5","beats":0.25},{"pitch":"C5","beats":0.25},{"pitch":"G4","beats":0.5},{"pitch":"G4","beats":0.25},{"pitch":"G4","beats":0.25},{"pitch":"E4","beats":0.5},{"pitch":"E4","beats":0.25},{"pitch":"E4","beats":0.25},{"pitch":"C4","beats":0.5},{"pitch":"C4","beats":0.25},{"pitch":"C4","beats":0.25},{"pitch":"G4","beats":0.75},{"pitch":"F4","beats":0.25},{"pitch":"E4","beats":0.75},{"pitch":"D4","beats":0.25},{"pitch":"C4","beats":2}]}]}
```

## Example 3: a spoken line, then a sung answer

```json
{"title":"Hello","bpm":96,"voices":[
{"name":"host","body":"male","notes":[{"say":"Here is our song.","beats":4}]},
{"name":"singer","body":"female","start":4,"notes":[{"pitch":"G4","beats":1,"lyric":"Hel-"},{"pitch":"A4","beats":1,"lyric":"lo"},{"pitch":"B4","beats":1,"lyric":"world,"},{"pitch":"C5","beats":1,"lyric":"hel-"},{"pitch":"B4","beats":1,"lyric":"lo"},{"pitch":"G4","beats":3,"lyric":"you."}]}]}
```

## Matching the original chimera-vocal-synth

Only when the user asks for the exact output of the original chimera-vocal-synth: every voice starts at beat 0
(no `start`, no leading rest), `lead_in` stays at its default, every voice gets `"seed": 0`, the score sets
`"normalize": false`, and only `bpm`, `body`, `vibrato`, notes and lyrics are used. Each voice then matches the
original's `render notes` sample for sample.

Otherwise leave `seed` out. Voices that enter later are rendered from shortly before their entry, and voices after
the first get their own random seed - both on purpose: timing stays the same, the sound is a different random take
(within the variation the original shows between its own seeds), and equal voices do not drift in lockstep. Spoken
replies also deliberately start each sentence at the speaking pitch instead of the original's glide up from 110 Hz.

## Limits

- English words only (other languages are read with English rules). One melody line per voice; one tempo per score.
- Up to 8 voices and 300 s of music per score. Rendering takes roughly 0.4 s per second of each voice with the
  numba accelerator and about 5 s per second of each voice without it (spread over several CPU cores).
- Only render lyrics that are in the public domain or written by you or the user; do not render complete
  copyrighted song lyrics.

## When the check fails

- "is not a note name": write C, C#, Db ... B plus an octave (C4 = middle C) or a MIDI number.
- "N syllables for M sounding notes": give each sounding note exactly one syllable; split words with `-`
  (`"mer-", "ri-", "ly"`), hold a vowel with `"-"`.
- "above/below the natural range": move the part an octave or choose another body.
- "spoken text needs N beats": raise `beats` of the `say` item or shorten the text.

## Helping the user set chimera up

Do not change settings yourself; explain the steps:
- Spoken replies with chimera: desktop app Capabilities -> Tools -> Text-to-Speech -> Chimera (local vocal synth);
  CLI `hermes config set tts.provider chimera`.
- Choose the voice: Settings -> Plugins -> chimera -> voice, or the chat command `/chimera voice female`
  (`/chimera voices` lists them).
- After installing chimera (or its accelerator), restart Hermes.
- Faster rendering: install the companion plugin chimera-numba with the id shown by `/chimera status`
  (normally `'GottZ/hermes-chimera#accel'`), then restart Hermes and run `/chimera warmup` once.
