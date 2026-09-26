#!/usr/bin/env bash
# Runs mix commands for the Jev routing prototype inside the hexpm/elixir image.
# Usage: ./run.sh <project-dir> <mix args...>     e.g. ./run.sh typesafe_client test
#        ./run.sh . test          (jev_routing project)
#        ./run.sh . bench         (alias for the bench)
# Keys are read from ~/.typesafe_key and ~/.anthropic_key into the container env; nothing is echoed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PROJ="${1:?project dir}"; shift
IMAGE="hexpm/elixir:1.20.4-erlang-28.5.0.7-debian-bookworm-20260918-slim"
BASE="experiments/jev_routing/elixir"
if [ "$PROJ" = "." ]; then REL="$BASE"; else REL="$BASE/${PROJ#./}"; fi
TS_KEY="$(cat "$HOME/.typesafe_key" 2>/dev/null || true)"
AN_KEY="$(cat "$HOME/.anthropic_key" 2>/dev/null || true)"
exec sudo -n docker run --rm \
  -v "$REPO:/work" -w "/work/$REL" \
  -v jev_routing_deps:/work/$BASE/deps \
  -v jev_routing_build:/work/$BASE/_build \
  -v jev_routing_tc_deps:/work/$BASE/typesafe_client/deps \
  -v jev_routing_tc_build:/work/$BASE/typesafe_client/_build \
  -v jev_routing_mix:/root/.mix \
  -v jev_routing_hex:/root/.hex \
  -e TYPESAFE_API_KEY="$TS_KEY" -e ANTHROPIC_API_KEY="$AN_KEY" -e MIX_ENV="${MIX_ENV:-dev}" \
  "$IMAGE" sh -c 'mix local.hex --force >/dev/null && mix local.rebar --force >/dev/null && mix deps.get >/dev/null && mix "$@"' sh "$@"
