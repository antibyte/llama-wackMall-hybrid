#!/usr/bin/env bash
# Alias for startspark.sh (Spark-X2.5-4B on GTX 1660 Ti).
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/startspark.sh" "$@"
