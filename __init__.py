"""Hermes plugin entry point for chimera (C5 owns this file; INTERFACES §1, §15).

Imports nothing heavy: numpy and the engine load lazily inside the handlers.
"""


def register(ctx):
    from .chimera_hermes import register as _register

    return _register(ctx)
