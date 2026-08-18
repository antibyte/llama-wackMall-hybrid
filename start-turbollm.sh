#!/usr/bin/env bash
# Start TurboLLM with the GTX 1660 Ti and GTX 1080 hybrid engines registered.
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/tools/turbollm/start.sh" "$@"
