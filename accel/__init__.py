"""chimera-numba: companion plugin that only installs numba for the chimera plugin.

Hermes requires a loadable module; this plugin registers nothing. The chimera plugin detects
numba with importlib.util.find_spec and uses it only inside its render worker processes.
"""


def register(ctx):  # noqa: ARG001 - Hermes plugin entry point
    """Nothing to register: numba is the whole point of this plugin."""
    return None
