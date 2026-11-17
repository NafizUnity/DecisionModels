#!/usr/bin/env bash
# BIODECISION: the shared realdata-med environment plus this model's checkpoints (see setup_med.sh).
MED_MODEL=BIODECISION exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_med.sh" "$@"
