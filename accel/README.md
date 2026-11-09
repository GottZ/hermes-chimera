# chimera-numba

Optional accelerator for the `chimera` Hermes plugin. It registers nothing; installing it makes
Hermes install `numba`, which chimera then uses in its render worker processes (about 12x faster
than the numpy fallback).

    hermes plugins install <owner>/<repo>#accel --yes-deps --enable

Restart Hermes afterwards. Platforms without numba wheels (macOS x86_64, musl Linux) keep using the
numpy backend of chimera.
