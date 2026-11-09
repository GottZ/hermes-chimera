"""``python -m chimera_vocal say|sing|check|selftest|bench`` - command line for the engine."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from typing import Sequence

COMMANDS = ("say", "sing", "check", "selftest", "bench")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m chimera_vocal", description="chimera vocal synth (port of "
                                "chimera-vocal-synth): speak English text, sing scores, check the backend.")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="{say,sing,check,selftest,bench}")
    s = sub.add_parser("say", help="speak text into a WAV file")
    s.add_argument("text")
    s.add_argument("--voice", default="male", help="preset or voice spec, e.g. female;pitch=230")
    s.add_argument("--out", default="say.wav")
    s.add_argument("--faithful", action="store_true",
                   help="render exactly like the template's `chimera render text` (no text preparation, one piece)")
    g = sub.add_parser("sing", help="render a score (JSON file) into WAV (convert to MP3 with ffmpeg yourself)")
    g.add_argument("score")
    g.add_argument("--out-dir", default=".")
    g.add_argument("--stem", default="score")
    c = sub.add_parser("check", help="validate a score (JSON file); exit 2 when it has errors")
    c.add_argument("score")
    sub.add_parser("selftest", help="render the built-in reference and compare (runs in a worker)")
    b = sub.add_parser("bench", help="real-time factor of a held sung vowel (one voice, the score path)")
    b.add_argument("--seconds", type=float, default=1.0, help="length of the held vowel (0.25-60 s)")
    for x in (s, g, b, c):
        x.add_argument("--backend", choices=("auto", "numba", "numpy"), default="auto")
        x.add_argument("--workers", type=int, default=0)
    return p


def _read_score(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from . import Engine, EngineConfig
    eng = Engine(EngineConfig(backend=getattr(args, "backend", "auto"), workers=getattr(args, "workers", 0)))
    try:
        if args.cmd == "check":
            r = eng.check_score(_read_score(args.score))
            print(json.dumps({"ok": r.ok, "issues": [dataclasses.asdict(i) for i in r.issues],
                              "seconds": r.seconds, "voice_seconds": r.voice_seconds, "lead_in": r.lead_in,
                              "estimate_s": r.estimate_s,
                              "voices": [dataclasses.asdict(v) for v in r.voices]}, indent=1, ensure_ascii=False))
            return 0 if r.ok else 2
        if args.cmd == "say":
            res = eng.speak_to_wav(args.text, args.out, args.voice, faithful=args.faithful)
            print(f"{args.out}: {res.seconds:.2f} s ({res.kernel})")
            return 0
        if args.cmd == "sing":
            res = eng.render_score(_read_score(args.score), args.out_dir, args.stem)
            print(json.dumps({"wav": res.path_wav, "seconds": res.seconds, "kernel": res.kernel,
                              "render_s": round(res.render_s, 3), "warnings": res.warnings}, indent=1))
            return 0
        if args.cmd == "selftest":
            print(eng.selftest())
            return 0
        if args.cmd == "bench":
            seconds = min(60.0, max(0.25, float(args.seconds)))
            score = {"bpm": 60, "lead_in": 0.0, "voices": [{"body": "male", "vibrato": 0,
                                                             "notes": [{"pitch": "A2", "beats": seconds,
                                                                        "lyric": "ah"}]}]}
            with tempfile.TemporaryDirectory(prefix="chimera-bench-") as d:
                res = eng.render_score(score, d, "bench")
            print(json.dumps({"kernel": res.kernel, "vowel_s": seconds, "audio_s": round(res.seconds, 3),
                              "wall_s": round(res.render_s, 3), "rtf": round(res.render_s / max(res.seconds, 1e-9), 3),
                              "status": str(eng.status())}))
            return 0
        return 2
    finally:
        eng.shutdown()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
