"""Use a checkpoint from a shared Hugging Face cache (models--ORG--NAME/snapshots/COMMIT) instead of downloading it.

  link      find a snapshot whose files match the pinned checkpoint, verify it, symlink PROJECT/FOLDER -> snapshot
  check     read-only: is PROJECT/FOLDER already the pinned checkpoint (used by realdata_setup.sh to skip work)
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


def model_folder(cache, repo):
    """The cache folder of REPO. Hub repo ids are case-insensitive, but a cache folder keeps the spelling used at
    download (e.g. models--google--gemma-4-31B-it for google/gemma-4-31b-it): an exact match first, else the one folder
    whose name differs only in case. The files are still verified against the pinned revision."""
    exact = cache / ("models--" + repo.replace("/", "--"))
    if exact.is_dir() or not cache.is_dir():
        return exact
    same = [d for d in cache.glob("models--*") if d.is_dir() and d.name.lower() == exact.name.lower()]
    if len(same) > 1:
        sys.exit(f"STOPPED: several cache folders match {repo} up to case: {', '.join(sorted(d.name for d in same))}")
    if same:
        print(f"{repo}: using cache folder {same[0].name} (same repo id, different capitalisation)", flush=True)
    return same[0] if same else exact


def cmd_link(args):
    manifest = json.loads(MANIFEST.read_text())
    entry = manifest["models"].get(args.repo)
    if entry is None:
        sys.exit(f"STOPPED: {args.repo} is not in {MANIFEST.name}")
    if entry["revision"] != args.revision:
        sys.exit(f"STOPPED: {MANIFEST.name} lists {args.repo}@{entry['revision'][:8]} but the lock pins {args.revision[:8]}; "
                 f"regenerate it with `python setup/shared_cache.py manifest`")
    cache = Path(args.cache).expanduser()
    model_dir = model_folder(cache, args.repo)
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


def cmd_check(args):
    """Read-only: is DEST the pinned checkpoint? A link must point to a snapshot of REPO in the cache (if given) whose files
    all match; a plain folder (a private download) is checked by file size only, which reads no weights. Exit 0 = ready."""
    entry = json.loads(MANIFEST.read_text())["models"].get(args.repo)
    if entry is None or entry["revision"] != args.revision:
        sys.exit(f"NOT READY: {MANIFEST.name} does not list {args.repo}@{args.revision[:8]}")
    dest = Path(args.dest)
    if not dest.exists():
        sys.exit(f"NOT READY: {dest} does not exist" + (" (broken link)" if dest.is_symlink() else ""))
    snap = Path(os.path.realpath(dest))
    if dest.is_symlink() and (args.cache or snap.parent.name == "snapshots"):   # a link into a Hugging Face cache
        if args.cache:
            want = model_folder(Path(args.cache).expanduser().resolve(), args.repo).resolve() / "snapshots"
            if snap.parent != want:
                sys.exit(f"NOT READY: {dest} links to {snap}, not to a snapshot in {want}")
        problems, _ = check_snapshot(snap, entry["files"])
        if problems:
            sys.exit(f"NOT READY: {dest} -> {snap}: {len(problems)} problem(s), first: {problems[0]}")
        print(f"ready: {dest} -> {snap} ({sum(not is_optional(f) for f in entry['files'])} files verified)")
        return
    if args.cache:
        sys.exit(f"NOT READY: {dest} is a plain folder, not a link into the shared cache {args.cache}")
    bad = [rel for rel, meta in entry["files"].items() if not is_optional(rel)
           and (not (dest / rel).is_file() or (dest / rel).stat().st_size != meta["size"])]
    if bad:
        sys.exit(f"NOT READY: {dest}: {len(bad)} file(s) missing or of the wrong size, first: {bad[0]}")
    print(f"ready: {dest} (plain folder; file sizes match, hashes not checked)")


def cmd_manifest(args):
    from huggingface_hub import HfApi
    lock = json.loads((HERE.parent / "revisions.lock.json").read_text())
    api, models = HfApi(), {}
    pins = [(m["repo"], m["revision"]) for m in lock["models"].values()]
    pins += [(m["base_repo"], m["base_revision"]) for m in lock["models"].values() if m.get("base_repo")]   # MedDecider bases
    for repo, revision in pins:
        info = api.model_info(repo, revision=revision, files_metadata=True)
        models[repo] = {"revision": revision, "files": {
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
    s = sub.add_parser("check", help="read-only: is DEST the pinned checkpoint (exit 0) or not (exit 1)")
    for a in ("--repo", "--revision", "--dest"):
        s.add_argument(a, required=True)
    s.add_argument("--cache", default=None, help="the shared cache DEST must link into")
    s.set_defaults(fn=cmd_check)
    sub.add_parser("manifest").set_defaults(fn=cmd_manifest)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
