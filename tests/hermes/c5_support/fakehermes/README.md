# fakehermes (Hermes binding tests only)

Minimal stand-ins for the public Hermes modules chimera_hermes imports (`hermes_cli/data/plugin_api.json`):
`agent.tts_provider`, `gateway.session_context`, `hermes_constants`, `hermes_cli.config`. Semantics follow
hermes-agent `0f0e96fc8e` (`agent/tts_provider.py`, `agent/provider_base.py`, `gateway/session_context.py:188-194`,
`hermes_constants.py:194,498-510`, `hermes_cli/config.py:1985`). State lives in module attributes so tests can
change it: `hermes_constants.HOME`, `hermes_cli.config.CONFIG`, `gateway.session_context.ENV`. Any write to core
configuration is recorded in `hermes_cli.config.WRITES` (the plugin must never write core config).
The `hermes`-marked tests run against the real Hermes instead.
