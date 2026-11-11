# Sourced by setup_*.sh after step()/die() are defined. Needs PROJECT_DIR, CHECKPOINT_REVISION, LOG_DIR.
#   get_checkpoint REPO FOLDER: with HF_SHARED_CACHE set (main.py setup-envs --hf-cache DIR), verify the shared copy and
#       link PROJECT_DIR/FOLDER to it; otherwise download the pinned revision into PROJECT_DIR/FOLDER.
#   $TESTED: setup/tested/<model>.txt, the pip freeze of the env that produced the recorded results (used as pip -c).
#   compare_tested: list every package that differs from $TESTED; stop if a core package differs.
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_HELPER="$SETUP_DIR/shared_cache.py"
MODEL_KEY="$(basename "${BASH_SOURCE[1]}" .sh)"; MODEL_KEY="${MODEL_KEY#setup_}"
TESTED="$SETUP_DIR/tested/$MODEL_KEY.txt"
[ -f "$TESTED" ] || die "tested package list $TESTED is missing"

get_checkpoint() {
    local repo="$1" folder="$2"
    if [ -n "${HF_SHARED_CACHE:-}" ]; then
        step "Using $repo @ ${CHECKPOINT_REVISION:0:8} from the shared cache $HF_SHARED_CACHE (no download)"
        python "$CHECKPOINT_HELPER" link --cache "$HF_SHARED_CACHE" --repo "$repo" --revision "$CHECKPOINT_REVISION" \
            --dest "$PROJECT_DIR/$folder" --record "$LOG_DIR/${folder}_source.json" \
            || die "the shared cache copy of $repo could not be used (details above)"
    else
        # never download through a link made by --hf-cache: that would write into the shared cache
        [ ! -L "$PROJECT_DIR/$folder" ] || die "$PROJECT_DIR/$folder links to $(readlink "$PROJECT_DIR/$folder"); rerun with --hf-cache, or delete the link to download a private copy"
        step "Downloading $repo @ ${CHECKPOINT_REVISION:0:8} into $PROJECT_DIR/$folder"
        hf download "$repo" --revision "$CHECKPOINT_REVISION" --local-dir "$PROJECT_DIR/$folder"
    fi
}

compare_tested() {
    step "Comparing installed packages with the tested versions ($TESTED)"
    python -m pip freeze > "$LOG_DIR/freeze_${MODEL_KEY}_installed.txt"
    python - "$TESTED" "$LOG_DIR/freeze_${MODEL_KEY}_installed.txt" <<'PY' || die "a core package differs from the tested version (see above)"
import re, sys
def pins(path):
    out = {}
    for line in open(path):
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)$", line.strip())
        if m:
            out[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
    return out
tested, installed = pins(sys.argv[1]), pins(sys.argv[2])
core = {"torch", "torchvision", "transformers", "flash-linear-attention", "fla-core", "causal-conv1d", "accelerate", "safetensors", "tokenizers"}
diff = sorted((n, v, installed.get(n, "missing")) for n, v in tested.items() if installed.get(n) != v)
for n, want, got in diff:
    print(f"  {'CORE ' if n in core else ''}{n}: tested {want}, installed {got}")
print(f"{len(tested) - len(diff)} of {len(tested)} tested packages match" + ("" if diff else " (all)"))
sys.exit(1 if any(n in core for n, _, _ in diff) else 0)
PY
}
