#!/usr/bin/env bash
# MEDDECIDER31: the shared realdata-med environment plus this model's checkpoints (see setup_med.sh).
MED_MODEL=MEDDECIDER31 exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_med.sh" "$@"
