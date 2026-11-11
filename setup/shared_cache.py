"""Use a checkpoint from a shared Hugging Face cache (models--ORG--NAME/snapshots/COMMIT) instead of downloading it.

  link      find a snapshot whose files match the pinned checkpoint, verify it, symlink PROJECT/FOLDER -> snapshot
  manifest  (maintainers) rewrite checkpoint_files.json from the Hub API for the revisions in revisions.lock.json

A snapshot is accepted if every file the model uses (everything except README, LICENSE, images, .eval_results and
.gitattributes) exists, is readable and has the pinned size and hash. Snapshot files are symlinks to blobs named by
their hash (LFS sha256, or the git blob sha1 for small files), so the check reads no weights; a file that is a plain copy
is hashed instead. The exact pinned commit is tried first; another commit is accepted only when it holds the same files
(on the Hub, Clef and Drex have newer commits that change only documentation). `link` needs only the standard library.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "checkpoint_files.json"
OPTIONAL_NAMES = {"README.md", "LICENSE", ".gitattributes"}
OPTIONAL_SUFFIXES = (".png", ".svg", ".jpg", ".jpeg", ".gif")


def is_optional(path):
    return path in OPTIONAL_NAMES or path.startswith((".eval_results/", "assets/")) or path.lower().endswith(OPTIONAL_SUFFIXES)


def file_hash(path, kind):
    if kind == "sha256":
        h = hashlib.sha256()
    else:                                           # git blob id: sha1 over "blob <size>\0" + content
        h = hashlib.sha1(f"blob {os.path.getsize(path)}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_snapshot(snap, files):
    """Problems for the required files (empty list = usable) and the optional files that are missing (information only)."""
    problems, optional_diff = [], []
    for rel, meta in sorted(files.items()):
        p = snap / rel
        optional = is_optional(rel)
        if not p.exists():                          # also false for a symlink to a missing or partial blob
            if optional:
                optional_diff.append(rel)
            else:
                problems.append(f"missing: {rel}")
            continue
        if optional:
            continue
        if not os.access(p, os.R_OK):
            problems.append(f"not readable: {rel}")
            continue
        size = p.stat().st_size
        if size != meta["size"]:
            problems.append(f"size {size} != pinned {meta['size']}: {rel}")
            continue
        if p.is_symlink() and Path(os.path.realpath(p)).parent.name == "blobs":
            found = Path(os.path.realpath(p)).name
            how = "blob name"
        else:
            print(f"  hashing {rel} ({size / 1e9:.1f} GB; plain file, not a cache symlink)", flush=True)
            found = file_hash(p, meta["hash_kind"])
            how = "content"
        if found != meta["hash"]:
            problems.append(f"{how} {found[:12]} != pinned {meta['hash'][:12]}: {rel}")
    return problems, optional_diff


def cmd_link(args):
    manifest = json.loads(MANIFEST.read_text())
    entry = manifest["models"].get(args.repo)
    if entry is None:
        sys.exit(f"STOPPED: {args.repo} is not in {MANIFEST.name}")
    if entry["revision"] != args.revision:
        sys.exit(f"STOPPED: {MANIFEST.name} lists {args.repo}@{entry['revision'][:8]} but the lock pins {args.revision[:8]}; "
                 f"regenerate it with `python setup/shared_cache.py manifest`")
    cache = Path(args.cache).expanduser()
    model_dir = cache / ("models--" + args.repo.replace("/", "--"))
    if not model_dir.is_dir():
        name = args.repo.split("/")[-1].lower()
        near = sorted(d.name for d in cache.glob("models--*") if name in d.name.lower()) if cache.is_dir() else []
        sys.exit(f"STOPPED: {model_dir} not found" + (f"; similar folders: {', '.join(near)}" if near else
                 ("" if cache.is_dir() else f" ({cache} is not a folder)")))
    snaps_dir = model_dir / "snapshots"
    snaps = sorted((d for d in snaps_dir.iterdir() if d.is_dir()), key=lambda d: (d.name != args.revision, -d.stat().st_mtime)) \
        if snaps_dir.is_dir() else []
    refs = {r.name: r.read_text().strip()[:8] for r in (model_dir / "refs").glob("*")} if (model_dir / "refs").is_dir() else {}
    print(f"{args.repo}: {len(snaps)} snapshot(s) in {model_dir} (refs: {refs or 'none'}); pinned {args.revision[:8]}", flush=True)
    report = {}
    for snap in snaps:
        problems, optional_diff = check_snapshot(snap, entry["files"])
        report[snap.name] = problems
        if problems:
            continue
        dest = Path(args.dest)
        if dest.is_symlink():
            dest.unlink()
        elif dest.exists():
            sys.exit(f"STOPPED: {dest} already exists as a real folder (an earlier download?). Move or delete it to use the "
                     f"shared cache, or rerun setup-envs without --hf-cache to keep using it.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(snap.resolve(), target_is_directory=True)
        same = snap.name == args.revision
        record = {"repo": args.repo, "pinned_revision": args.revision, "snapshot_commit": snap.name, "snapshot": str(snap.resolve()),
                  "exact_pinned_commit": same, "required_files_verified": sum(not is_optional(f) for f in entry["files"]),
                  "optional_files_missing": optional_diff, "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if args.record:
            Path(args.record).parent.mkdir(parents=True, exist_ok=True)
            Path(args.record).write_text(json.dumps(record, indent=2))
        print(f"OK: {dest} -> {snap.resolve()}", flush=True)
        print(f"    commit {snap.name[:8]} ({'the pinned commit' if same else 'not the pinned commit, but every file the model uses matches it'}); "
              f"{record['required_files_verified']} files verified" + (f"; optional files missing: {', '.join(optional_diff)}" if optional_diff else ""), flush=True)
        return
    for name, problems in report.items():
        print(f"  snapshot {name[:8]}: {len(problems)} problem(s)", flush=True)
        for line in problems[:10]:
            print(f"    {line}", flush=True)
    sys.exit(f"STOPPED: no snapshot of {args.repo} in the shared cache matches the pinned checkpoint. Ask the admin to complete it "
             f"(hf download {args.repo} --revision {args.revision}), or rerun setup-envs without --hf-cache to download it.")


def cmd_manifest(args):
    from huggingface_hub import HfApi
    lock = json.loads((HERE.parent / "revisions.lock.json").read_text())
    api, models = HfApi(), {}
    for m in lock["models"].values():
        info = api.model_info(m["repo"], revision=m["revision"], files_metadata=True)
        models[m["repo"]] = {"revision": m["revision"], "files": {
            s.rfilename: ({"size": s.size, "hash": s.lfs.sha256, "hash_kind": "sha256"} if s.lfs else
                          {"size": s.size, "hash": s.blob_id, "hash_kind": "git_blob_sha1"}) for s in info.siblings}}
    MANIFEST.write_text(json.dumps({"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": models}, indent=1))
    counts = ", ".join("%s: %d files" % (r, len(v["files"])) for r, v in models.items())
    print(f"wrote {MANIFEST} ({counts})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("link")
    for a in ("--cache", "--repo", "--revision", "--dest"):
        s.add_argument(a, required=True)
    s.add_argument("--record", default=None, help="write a JSON record of what was linked and verified")
    s.set_defaults(fn=cmd_link)
    sub.add_parser("manifest").set_defaults(fn=cmd_manifest)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
