"""Build the public release v2.1.0 of dcfnet-ds (IEEE Access Access-2026-52475).

Copies only; nothing in the working folder is moved or changed.

Steps
  1. python -m src.p1_verify_resub_v2 must pass (skip with --skip-verify).
  2. Select the public files (rules below, following .gitignore and p1_release_map_v1.md).
  3. Write
       _release_v2.1.0/snapshot/dcfnet-ds-v2.1.0/...   complete public tree
       _release_v2.1.0/dcfnet-ds_v2.1.0.zip            the same tree, for Zenodo
       _release_v2.1.0/MANIFEST_sha256.txt             SHA-256 of every file in the tree
       _release_v2.1.0/upload/batchN/...               files new or changed since v2.0.1,
                                                       90 per batch, for the GitHub web upload
       _release_v2.1.0/CHANGES_v2.1.0.txt              which files are new / changed / removed / unchanged
       _release_v2.1.0/REMOVE_v2.1.0.txt               files to delete from the GitHub repository

"Changed since v2.0.1": a file listed in _release_v2.0.1_full/MANIFEST_sha256.txt whose SHA-256
differs, or a file not listed there (uploading an unchanged file again changes nothing on GitHub).
Files listed there but absent from v2.1.0 are written to REMOVE_v2.1.0.txt, to be deleted on GitHub.

Run
  python -m src.p1_build_pub_v1
  python -m src.p1_build_pub_v1 --skip-verify
"""

import argparse
import fnmatch
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from src import config as C

VERSION = "v2.1.0"
BATCH = 90
ROOT = Path(C.ROOT)
OUT = ROOT / f"_release_{VERSION}"
PREV_MANIFEST = ROOT / "_release_v2.0.1_full" / "MANIFEST_sha256.txt"

TOP_FILES = [
    "README.md", "README_notice_v2.0.1.md", "LICENSE",
    "REPRODUCE_paper1_resubmission.md", "REPRODUCE_paper1_scaling.md", "REPRODUCE_paper2_degree.md",
    "requirements.txt", "requirements-lock.txt", "environment-lock.yml",
    ".gitignore", ".gitattributes",
    "main.py", "run_all.py", "run_experiment.py",
    "run_1a.bat", "run_1b.bat", "run_1c.bat", "run_1d.bat",
    "run_1e.bat", "run_1f.bat", "run_1g.bat",
]
SRC_EXCLUDE = {"check_perclass.py", "check_vanish.py", "p1_fill_placeholders_v1.py"}
TREES = [  # (folder, file patterns)
    ("outputs/tables", ["*.csv", "*.xlsx"]),
    ("outputs/paper2_degree", ["*"]),
    ("outputs/figures/paper1_scaler", ["*_resub.*"]),
]
# .gitignore scratch rules that apply inside outputs/
IGNORE = ["*.backup.csv", "*.bak*.csv", "*_backup.csv", "_tmp_*", "*.log", "*.source.json"]
IGNORE_DIRS = {"_archive_1e_before", "__pycache__"}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ignored(rel):
    parts = rel.parts
    if any(d in IGNORE_DIRS or d.startswith("_archive_") for d in parts[:-1]):
        return True
    return any(fnmatch.fnmatch(parts[-1], pat) for pat in IGNORE)


def select():
    files = []
    for name in TOP_FILES:
        p = ROOT / name
        if not p.is_file():
            sys.exit(f"[stop] {name} missing")
        files.append(p)
    files += [p for p in sorted((ROOT / "src").glob("*.py")) if p.name not in SRC_EXCLUDE]
    for folder, pats in TREES:
        d = ROOT / folder
        if not d.is_dir():
            sys.exit(f"[stop] {folder} missing")
        for pat in pats:
            for p in sorted(d.rglob(pat)):
                if p.is_file() and not ignored(p.relative_to(ROOT)):
                    files.append(p)
    return sorted(set(files))


def previous():
    if not PREV_MANIFEST.exists():
        sys.exit(f"[stop] {PREV_MANIFEST} missing")
    prev = {}
    for line in PREV_MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            prev[rel] = h
    return prev, PREV_MANIFEST.stat().st_mtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-verify", action="store_true")
    a = ap.parse_args()

    if OUT.exists():
        sys.exit(f"[stop] {OUT.name} already exists. Delete it or rename it (nothing is overwritten).")
    if not a.skip_verify:
        print("[verify] python -m src.p1_verify_resub_v2")
        env = dict(os.environ, PYTHONUTF8="1")
        r = subprocess.run([sys.executable, "-m", "src.p1_verify_resub_v2"], cwd=ROOT, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        tail = [l for l in r.stdout.splitlines() if l.strip()][-1:] or ["(no output)"]
        print("  ", tail[0])
        if r.returncode != 0:
            sys.exit("[stop] verification did not pass; run it directly to see the failing items.")

    files = select()
    prev, _ = previous()
    snap = OUT / "snapshot" / f"dcfnet-ds-{VERSION}"
    man, new, changed, same = [], [], [], []
    for p in files:
        rel = p.relative_to(ROOT)
        key = rel.as_posix()
        dst = snap / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        h = sha(p)
        man.append(f"{h}  {key}")
        if key in prev:
            (same if prev[key] == h else changed).append(key)
        else:
            new.append(key)
    keys = {p.relative_to(ROOT).as_posix() for p in files}
    removed = sorted(k for k in prev if k not in keys and k != "MANIFEST_sha256.txt")

    (OUT / "MANIFEST_sha256.txt").write_text("\n".join(man) + "\n", encoding="utf-8")
    shutil.copy2(OUT / "MANIFEST_sha256.txt", snap / "MANIFEST_sha256.txt")

    upload = sorted(new + changed) + ["MANIFEST_sha256.txt"]
    for i, key in enumerate(upload):
        dst = OUT / "upload" / f"batch{i // BATCH + 1}" / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snap / key, dst)

    zpath = OUT / f"dcfnet-ds_{VERSION}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(snap.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(snap.parent))

    lines = [f"dcfnet-ds {VERSION}: {len(files)} files + MANIFEST",
             f"compared with {PREV_MANIFEST.relative_to(ROOT).as_posix()}", ""]
    for title, group in [("NEW (not in the v2.0.1 manifest)", new), ("CHANGED", changed),
                         ("REMOVED (delete on GitHub)", removed)]:
        lines.append(f"[{title}] {len(group)}")
        lines += [f"  {k}" for k in sorted(group)]
        lines.append("")
    lines.append(f"[UNCHANGED] {len(same)}")
    (OUT / f"CHANGES_{VERSION}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / f"REMOVE_{VERSION}.txt").write_text("\n".join(removed) + "\n", encoding="utf-8")

    print(f"\n[done] {OUT}")
    print(f"  snapshot  {len(files) + 1} files")
    print(f"  zip       {zpath.name}  {zpath.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  new {len(new)}, changed {len(changed)}, removed {len(removed)}, unchanged {len(same)}")
    for b in sorted((OUT / "upload").glob("batch*")):
        print(f"  upload/{b.name}: {sum(1 for x in b.rglob('*') if x.is_file())} files")
    print(f"  list      CHANGES_{VERSION}.txt\n")


if __name__ == "__main__":
    main()
