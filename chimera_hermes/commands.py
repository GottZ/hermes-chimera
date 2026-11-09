"""``/chimera`` (slash command) and ``hermes chimera`` (CLI).

Read-only towards Hermes' own configuration: the only write is the plugin's own setting ``voice`` through
``ctx.set_config``. Making chimera the text-to-speech provider stays the user's step (desktop: Capabilities -> Tools
-> Text-to-Speech; CLI: ``hermes config set tts.provider chimera``); the status text explains it.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
from typing import Any, Callable

from . import _vocal, media, paths, settings

SLASH_DESCRIPTION = "Chimera voices: status, voices, voice <id>, warmup, test"
SLASH_ARGS_HINT = "[status|voices|voice <id>|warmup|test]"
CLI_HELP = "Chimera vocal synth"
COMPANION = "chimera-numba"
DESKTOP_INSTALL = "Desktop: Capabilities -> Plugins -> Install from repository"
DESKTOP_TTS = "Desktop: Capabilities -> Tools -> Text-to-Speech -> Chimera (local vocal synth)"


DEFAULT_SOURCE = "GottZ/hermes-chimera"


def _plugin_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_SCP_LIKE = re.compile(r"^[^@/:]+:[^@/]*@([^@/:]+:.*)$")      # user:secret@host:path (scp-like)


def _clean_source(url: str) -> str:
    """Install source without credentials, query or fragment - it is shown in chat and handed to the model. Any URL
    with a host loses its user info (``ssh://git:token@…``, ``git+https://user:token@…``); an scp-like
    ``user:secret@host:path`` loses the part before ``@``; ``github.com/<o>/<r>(.git)`` over http(s) is shortened to
    ``<o>/<r>``."""
    url = url.strip()
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return "(install source not shown: unparsable URL)"
    if parts.scheme and host:
        netloc = (f"[{host}]" if ":" in host else host) + (f":{port}" if port else "")
        path = parts.path.rstrip("/")
        if host.lower() == "github.com" and parts.scheme in ("http", "https", "git+http", "git+https"):
            short = path.strip("/")
            short = short[:-4] if short.endswith(".git") else short
            if short.count("/") == 1:
                return short
        return urllib.parse.urlunsplit((parts.scheme, netloc, path, "", ""))
    bare = url.split("#", 1)[0].split("?", 1)[0]
    m = _SCP_LIKE.match(bare)
    return m.group(1) if m else bare


def _install_record_source(root: str) -> str | None:
    """Source Hermes recorded when it installed this plugin (``<plugins dir>/.install-metadata.json``)."""
    try:
        with open(os.path.join(os.path.dirname(root), ".install-metadata.json"), encoding="utf-8-sig") as f:
            rows = json.load(f)
        row = rows.get(os.path.basename(root)) if isinstance(rows, dict) else None
        src = row.get("source") if isinstance(row, dict) else None
        return src if isinstance(src, str) and src.strip() else None
    except (OSError, ValueError):
        return None


def _git_origin(root: str) -> str | None:
    """``remote "origin"`` URL from the plugin's own ``.git/config`` (no git process)."""
    try:
        with open(os.path.join(root, ".git", "config"), encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    in_origin = False
    for line in lines:
        s = line.strip()
        if s.startswith("["):
            in_origin = s.replace(" ", "") == '[remote"origin"]'
        elif in_origin and s.startswith("url"):
            url = s.split("=", 1)[1].strip()
            return url or None
    return None


def install_source() -> str:
    """Where chimera came from, for the companion's install id ``<source>#accel``: Hermes' install record, else the
    plugin's git origin, else the published repository."""
    root = _plugin_root()
    for src in (_install_record_source(root), _git_origin(root)):
        if src:
            cleaned = _clean_source(src)
            if cleaned:
                return cleaned
    return DEFAULT_SOURCE


def companion_on_disk() -> bool:
    """chimera-numba is installed next to chimera in Hermes' plugins directory (its directory, or a row in Hermes'
    install record). A running Hermes process never switches to the dependency environment an install commits, so
    numba can be installed and still not importable here until Hermes restarts."""
    plugins_dir = os.path.dirname(_plugin_root())
    if os.path.isfile(os.path.join(plugins_dir, COMPANION, "plugin.yaml")):
        return True
    try:
        with open(os.path.join(plugins_dir, ".install-metadata.json"), encoding="utf-8-sig") as f:
            rows = json.load(f)
        return isinstance(rows, dict) and isinstance(rows.get(COMPANION), dict)
    except (OSError, ValueError):
        return False


RESTART_FOR_COMPANION = (f"{COMPANION} is installed, but this Hermes process started before it - restart Hermes "
                         "(hermes gateway restart, or quit and reopen the desktop app) to render with numba")


def numba_missing_line() -> str:
    if companion_on_disk():
        return "numba: not importable in this process - " + RESTART_FOR_COMPANION
    return "numba: not installed - " + companion_hint()


def companion_hint() -> str:
    src = install_source()
    return (f"install the companion plugin {COMPANION} for about 19x faster rendering: "
            f"hermes plugins install '{src}#accel' --yes-deps --enable ({DESKTOP_INSTALL}, same id), "
            "then restart Hermes")


def _provider_line() -> str:
    provider = settings.core_tts().get("provider")
    if isinstance(provider, str) and provider.strip().lower() == "chimera":
        return "tts.provider: chimera (spoken replies use chimera)"
    shown = provider if isinstance(provider, str) and provider else "(Hermes default)"
    return (f"tts.provider: {shown} - to read replies with chimera: {DESKTOP_TTS}, or run "
            "`hermes config set tts.provider chimera`")


def _voice_line(ctx: Any) -> str:
    try:
        tts_voice = settings.core_tts().get("voice")
        preset = settings.resolve_voice(ctx, tts_voice if isinstance(tts_voice, str) else None)
        return f"voice: {getattr(preset, 'id', preset)}"
    except Exception as e:
        return f"voice: unavailable ({type(e).__name__})"


def status_text(ctx: Any) -> str:
    lines = ["chimera status"]
    numpy_ok = _vocal.numpy_ok()
    if not numpy_ok:
        lines.append("numpy: NOT importable in this Hermes process - if chimera was just installed, restart Hermes; "
                     "otherwise reinstall chimera with its dependencies (hermes plugins install <source> --yes-deps)")
    else:
        lines.append("numpy: available")
    st = None
    if numpy_ok:
        try:
            st = _vocal.engine(ctx).status()
        except Exception as e:
            lines.append(f"engine: unavailable ({type(e).__name__}: {e})")
    if st is not None:
        if st.numba_installed:
            state = st.numba_state
            line = f"numba: installed{' ' + st.numba_version if st.numba_version else ''}, state {state}"
            if state == "failed":
                line += f" ({st.numba_error or 'unknown error'}) - rendering with numpy"
            elif state in ("absent", "cold"):
                if st.numba_error:
                    line += f" ({st.numba_error})"
                line += " - run /chimera warmup (or `hermes chimera warmup`) to compile the kernels once"
            if st.marker:
                line += f", warm-up marker {str(st.marker.get('content_hash', ''))[:16]}"
            lines.append(line)
        else:
            lines.append(numba_missing_line())
        if st.restart_needed:
            lines.append("restart Hermes: packages were installed after this process started")
        lines.append(f"workers: {st.workers} ({st.workers_alive} running); cache: {st.cache_dir or paths.numba_cache_dir()}")
    elif numpy_ok is False:
        lines.append("numba: installed" if _vocal.numba_installed() else numba_missing_line())
    ff = media.ffmpeg_path()
    lines.append(f"ffmpeg: {ff}" if ff else "ffmpeg: not found (music is delivered as WAV)")
    lines.append(_provider_line())
    lines.append(_voice_line(ctx))
    if os.environ.get("HERMES_PLUGIN_HOST_PROCESS") == "1":
        lines.append("streaming: off (plugins.isolation: host) - replies are spoken sentence by sentence")
    return "\n".join(lines)


def voices_text() -> str:
    rows = ["id | voice | gender"]
    for v in _vocal.api().presets.list_voices():
        rows.append(f"{v.get('id')} | {v.get('display', v.get('id'))} | {v.get('gender', '')}")
    rows.append("Use /chimera voice <id> (or a spec like 'female;pitch=230').")
    return "\n".join(rows)


def _set_voice(ctx: Any, spec: str) -> str:
    spec = spec.strip()
    presets = _vocal.api().presets
    if not spec:
        return _voice_line(ctx) + "\nUsage: /chimera voice <id> - " + ", ".join(presets.PRESETS)
    if presets.parse_voice_spec(spec) is None:
        return f"unknown voice {spec!r}: choose one of {', '.join(presets.PRESETS)} or a spec like 'female;pitch=230'"
    if ctx is None:
        return "cannot store the voice: no plugin context"
    ctx.set_config("voice", spec)
    out = f"chimera voice set to {spec}."
    provider = settings.core_tts().get("provider")
    if not (isinstance(provider, str) and provider.strip().lower() == "chimera"):
        out += (f"\nchimera is not your text-to-speech provider yet: {DESKTOP_TTS}, or run "
                "`hermes config set tts.provider chimera`.")
    return out


def _warmup(ctx: Any) -> str:
    if not _vocal.numpy_ok():
        return _vocal.NUMPY_MISSING
    try:
        eng = _vocal.engine(ctx)
        eng.warmup()
        st = eng.status()
    except Exception as e:
        return f"warm-up failed: {type(e).__name__}: {e} - chimera keeps rendering with numpy"
    if st.numba_state == "failed":
        return f"warm-up failed: {st.numba_error or 'unknown error'} - chimera keeps rendering with numpy"
    if getattr(getattr(eng, "scheduler", None), "backend", None) == "numpy":
        return "the plugin setting backend is numpy: numba is not used (set backend to auto to use it)"
    if not st.numba_installed:
        return "warm-up started (numpy backend). " + (RESTART_FOR_COMPANION if companion_on_disk() else
                                                       companion_hint())
    return f"warm-up started: numba state {st.numba_state} (see /chimera status)"


def _selftest(ctx: Any) -> str:
    if not _vocal.numpy_ok():
        return _vocal.NUMPY_MISSING
    try:
        r = _vocal.engine(ctx).selftest()
    except Exception as e:
        return f"self-test failed: {type(e).__name__}: {e}"
    verdict = "passed" if getattr(r, "passed", False) else "FAILED"
    return (f"self-test {verdict}: backend {getattr(r, 'backend', '?')}, max rel. error "
            f"{getattr(r, 'max_rel_err', float('nan')):.3g}, {getattr(r, 'seconds', 0.0):.2f} s "
            f"{getattr(r, 'detail', '')}").strip()


def handle_slash(raw_args: str, *, ctx: Any = None) -> str:
    """``/chimera [status|voices|voice <id>|warmup|test]``; never raises."""
    ctx = ctx if ctx is not None else _vocal.ctx()
    parts = (raw_args or "").strip().split(maxsplit=1)
    sub = parts[0].lower() if parts else "status"
    rest = parts[1] if len(parts) > 1 else ""
    try:
        if sub == "status":
            return status_text(ctx)
        if sub == "voices":
            return voices_text()
        if sub == "voice":
            return _set_voice(ctx, rest)
        if sub == "warmup":
            return _warmup(ctx)
        if sub in ("test", "selftest"):
            return _selftest(ctx)
        return f"unknown subcommand {sub!r}. Usage: /chimera {SLASH_ARGS_HINT}"
    except Exception as e:  # a slash command answers, it does not crash the session
        return f"/chimera {sub} failed: {type(e).__name__}: {e}"


def handle_slash_factory(ctx: Any) -> Callable[[str], str]:
    def chimera(raw_args: str) -> str:
        return handle_slash(raw_args, ctx=ctx)

    return chimera


# ------------------------------------------------------------------------------------------------ CLI

def setup_cli(subparser: Any) -> None:
    sub = subparser.add_subparsers(dest="chimera_cmd")
    sub.add_parser("status", help="numpy/numba/ffmpeg state, provider, voice")
    sub.add_parser("voices", help="list the voice presets")
    sub.add_parser("warmup", help="compile the numba kernels once (needs chimera-numba)")
    sub.add_parser("selftest", help="render a short reference and compare it")
    say = sub.add_parser("say", help="speak TEXT into a WAV file")
    say.add_argument("text", nargs="+")
    say.add_argument("--voice", default=None)
    say.add_argument("--out", default=None)
    sing = sub.add_parser("sing", help="render a score JSON file")
    sing.add_argument("score")
    sing.add_argument("--out", default=None)
    check = sub.add_parser("check", help="validate a score JSON file")
    check.add_argument("score")


def _read_score(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_cli(args: Any) -> int:
    ctx = _vocal.ctx()
    cmd = getattr(args, "chimera_cmd", None) or "status"
    out = sys.stdout
    try:
        if cmd == "status":
            print(status_text(ctx), file=out)
            return 0
        if cmd == "voices":
            print(voices_text(), file=out)
            return 0
        if not _vocal.numpy_ok():
            print(_vocal.NUMPY_MISSING, file=sys.stderr)
            return 2
        eng, cfg = _vocal.engine_and_config(ctx)
        if cmd == "warmup":
            eng.warmup()        # starts the warm-up at once: state "warming" until it ends ready, failed or cold
            deadline = time.monotonic() + 900.0
            while time.monotonic() < deadline:
                st = eng.status()
                if not st.numba_installed or st.numba_state != "warming":
                    break
                time.sleep(0.5)
            print(status_text(ctx), file=out)
            st = eng.status()
            return 1 if st.numba_state == "failed" else 0
        if cmd == "selftest":
            r = eng.selftest()
            print(_selftest(ctx) if r is None else
                  f"self-test {'passed' if r.passed else 'FAILED'}: backend {r.backend}, {r.detail}", file=out)
            return 0 if getattr(r, "passed", False) else 1
        if cmd == "check":
            report = eng.check_score(_read_score(args.score), max_voices=_vocal.score_options(cfg)["max_voices"])
            for i in report.issues:
                print(f"{i.severity}: {i.path}: {i.message}", file=out)
            print(f"{'ok' if report.ok else 'invalid'}: {report.seconds:.1f} s, {len(report.voices)} voice(s)", file=out)
            return 0 if report.ok else 1
        if cmd == "say":
            text = " ".join(args.text)
            if args.voice:
                spec, preset = args.voice.strip(), _vocal.api().presets.parse_voice_spec(args.voice.strip())
            else:
                tts_voice = settings.core_tts().get("voice")
                spec, preset = settings.resolve_voice_spec(ctx, tts_voice if isinstance(tts_voice, str) else None)
            if preset is None:
                print(f"unknown voice {args.voice!r}", file=sys.stderr)
                return 2
            if args.out:
                path = args.out
            else:
                d, stem = paths.new_output_stem(text[:60], "speech")
                path = str(d / f"{stem}.wav")
            eng.speak_to_wav(text, path, spec, speed=settings.resolve_speed(None, preset),
                             faithful=settings.faithful(ctx), tmp_dir=str(paths.tmp_dir(create=True)),
                             **_vocal.speech_options(cfg))
            print(path, file=out)
            return 0
        if cmd == "sing":
            score = _read_score(args.score)
            opts = _vocal.score_options(cfg)
            report = eng.check_score(score, max_voices=opts["max_voices"])
            if not report.ok:
                for i in report.issues:
                    print(f"{i.severity}: {i.path}: {i.message}", file=sys.stderr)
                return 1
            if args.out:
                out_dir, name = os.path.split(os.path.abspath(args.out))
                stem = name[:-4] if name.lower().endswith(".wav") else name
            else:
                d, stem = paths.new_output_stem(score.get("title") if isinstance(score, dict) else None, "score")
                out_dir = str(d)
            res = eng.render_score(score, out_dir, stem, want_mp3=False, tmp_dir=str(paths.tmp_dir(create=True)),
                                   **opts)
            print(res.path_wav, file=out)
            return 0
        print(f"unknown command {cmd!r}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"chimera {cmd} failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
