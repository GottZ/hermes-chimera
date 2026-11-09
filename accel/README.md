# chimera-numba

Optional accelerator for the `chimera` Hermes plugin. It registers nothing: installing it makes Hermes install
`numba`, which chimera then uses inside its render worker processes (about 19x faster than the numpy fallback).
chimera never imports numba into the Hermes process itself.

    hermes plugins install 'GottZ/hermes-chimera#accel' --yes-deps --enable

Keep the quotes: zsh with `extendedglob` treats `#` as a pattern character. `GottZ/hermes-chimera/accel` is the
same id without a `#`.
In the desktop app use Capabilities -> Plugins -> Install from repository with the same id. Restart Hermes
afterwards, then run `/chimera warmup` once to compile the kernels.

Platforms without numba wheels (macOS x86_64, musl Linux, Android) keep using chimera's numpy backend. Removing this
plugin does not remove numba from the environment; chimera detects numba itself, not this plugin. Updating: installed
from the repository as above, this plugin has no `.git` folder, so `hermes plugins update` skips it; reinstall with
`hermes plugins install 'GottZ/hermes-chimera#accel' --force --yes-deps --enable`.
