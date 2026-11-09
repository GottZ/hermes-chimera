"""``python -m chimera_vocal say|sing|check|selftest|bench`` - command line for the engine."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
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
    g = sub.add_parser("sing", help="render a score (JSON file) into WAV")
    g.add_argument("score")
    g.add_argument("--out-dir", default=".")
    g.add_argument("--stem", default="score")
    g.add_argument("--mp3", action="store_true", help="also write MP3 when ffmpeg is installed")
    c = sub.add_parser("check", help="validate a score (JSON file); exit 2 when it has errors")
    c.add_argument("score")
    sub.add_parser("selftest", help="render the built-in reference and compare (runs in a worker)")
    b = sub.add_parser("bench", help="real-time factor of 1 s of a held vowel")
    b.add_argument("--seconds", type=float, default=1.0)
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
            audio, kernel = eng.speak(args.text, args.voice)
            audio.to_wav(args.out)
            print(f"{args.out}: {audio.duration_s():.2f} s ({kernel})")
            return 0
        if args.cmd == "sing":
            res = eng.render_score(_read_score(args.score), args.out_dir, args.stem, want_mp3=args.mp3)
            print(json.dumps({"wav": res.path_wav, "mp3": res.path_mp3, "seconds": res.seconds, "kernel": res.kernel,
                              "render_s": round(res.render_s, 3), "warnings": res.warnings}, indent=1))
            return 0
        if args.cmd == "selftest":
            print(eng.selftest())
            return 0
        if args.cmd == "bench":
            t = time.perf_counter()
            audio, kernel = eng.speak("ah.", "male")
            dt = time.perf_counter() - t
            print(json.dumps({"kernel": kernel, "audio_s": audio.duration_s(), "wall_s": round(dt, 3),
                              "rtf": round(dt / max(audio.duration_s(), 1e-9), 3), "status": str(eng.status())}))
            return 0
        return 2
    finally:
        eng.shutdown()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
