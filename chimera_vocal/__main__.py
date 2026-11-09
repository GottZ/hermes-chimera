"""``python -m chimera_vocal say|sing|check|selftest|bench`` - command line for the engine (C4)."""

from __future__ import annotations

import sys
from typing import Sequence

COMMANDS = ("say", "sing", "check", "selftest", "bench")


def main(argv: Sequence[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
