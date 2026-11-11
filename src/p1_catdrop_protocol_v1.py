"""Separate the 40-epoch budget runs from the 42-feature main-protocol runs.

Why this exists
  Table IX of the manuscript has two 42-feature columns:
    "42 feat."          early stopping (patience 20, cap 500)  0.3677 / 0.4011 / 0.4013 / 0.3297
    "42 feat., 40 ep."  fixed 40 epochs, no early stopping       0.3675 / 0.3981 / 0.4027 / 0.3297
  The 40-epoch runs (src/config_old.py: EPOCHS = 40, EARLY_STOP_PATIENCE = 999)
  were trained after the main-protocol runs under the same names and replaced
  their checkpoints and seeds files. The main-protocol seed 1-3 values survive
  only in backup_catdrop/. run_1e.bat therefore re-evaluated 40-epoch
  checkpoints as seeds 1-3 and joined them to main-protocol seeds 4-10.

  step isolate : move the 40-epoch checkpoints and histories (seeds 1-3) to
                 */budget40/ and keep their seeds/summary files there, so that
                 seeds 1-3 can be retrained under the main protocol and Table IX
                 stays reproducible.
  step compare : after retraining, compare the new seeds 1-3 with backup_catdrop/
                 (the values printed in the manuscript) and report the change.

Run
  python -m src.p1_catdrop_protocol_v1 --step isolate
  python -m src.p1_catdrop_protocol_v1 --step compare
"""

import argparse
import shutil

import pandas as pd

from src import config as C

METRIC = "PR-AUC(macro)"
SCALERS = ["minmax", "signedlog", "signedlog_minmax", "standard"]
MAIN = ["minmax", "signedlog", "signedlog_minmax"]
SEED_SUFFIX = {1: "", 2: "_s2", 3: "_s3"}
BUDGET = 40
# Table IX "42 feat., 40 ep." column (three-seed means)
T9_BUDGET = {"signedlog_minmax": 0.4027, "signedlog": 0.3981, "minmax": 0.3675, "standard": 0.3297}

ROOT = C.TABLE_DIR.parent.parent
MODEL_DIR = C.TABLE_DIR.parent / "models"
ARCHIVE = C.TABLE_DIR / "_archive_1e_before"
BACKUP = ROOT / "backup_catdrop"
B_TAB = C.TABLE_DIR / "budget40"
B_MOD = MODEL_DIR / "budget40"


def n_epochs(path):
    return len(pd.read_csv(path)) if path.exists() else None


def isolate():
    if C.EPOCHS != 500 or C.EARLY_STOP_PATIENCE != 20:
        raise SystemExit(f"[stop] src/config.py is not the main protocol "
                         f"(EPOCHS={C.EPOCHS}, EARLY_STOP_PATIENCE={C.EARLY_STOP_PATIENCE})")
    B_TAB.mkdir(parents=True, exist_ok=True)
    B_MOD.mkdir(parents=True, exist_ok=True)

    print("\n[isolate] checkpoints and histories, seeds 1-3")
    for sc in SCALERS:
        for seed, suf in SEED_SUFFIX.items():
            name = f"nfunsw_catdrop_{sc}_m4{suf}"
            hist = C.TABLE_DIR / f"history_{name}.csv"
            ckpt = MODEL_DIR / f"{name}.keras"
            dest_h, dest_c = B_TAB / hist.name, B_MOD / ckpt.name
            if dest_h.exists():
                print(f"  [skip] {name}: already in budget40/")
                continue
            ep = n_epochs(hist)
            if ep != BUDGET:
                print(f"  [keep] {name}: history has {ep} epochs, not a {BUDGET}-epoch run")
                continue
            if not ckpt.exists():
                raise SystemExit(f"[stop] {ckpt.name} missing for a {BUDGET}-epoch history")
            shutil.move(str(hist), dest_h)
            shutil.move(str(ckpt), dest_c)
            print(f"  [move] {name}  ({ep} epochs)")

    print("\n[isolate] seeds and summary files of the 40-epoch runs")
    for sc in SCALERS:
        for kind in ["seeds", "summary"]:
            fn = f"{kind}_nfunsw_catdrop_{sc}.csv"
            dest = B_TAB / fn
            if dest.exists():
                continue
            src = (ARCHIVE / fn) if sc in MAIN else (C.TABLE_DIR / fn)
            if not src.exists():
                raise SystemExit(f"[stop] {src} missing")
            shutil.copy(src, dest)   # copy time is the reference for step compare
        d = pd.read_csv(B_TAB / f"seeds_nfunsw_catdrop_{sc}.csv")
        mu = d[METRIC].mean()
        ok = sorted(d.seed) == [1, 2, 3] and round(mu, 4) == T9_BUDGET[sc]
        print(f"  {'PASS' if ok else 'FAIL'}  budget40 {sc:17s} seeds {sorted(d.seed)} "
              f"mean {mu:.4f}   Table IX {T9_BUDGET[sc]}")
        if not ok:
            raise SystemExit("[stop] budget40 files do not match Table IX")
    print("\n  Seeds 1-3 of minmax, signedlog, signedlog_minmax can now be retrained.\n")


def compare():
    print(f"\n[compare] retrained seeds 1-3 vs backup_catdrop/ (manuscript Tables III, IX)")
    for sc in MAIN:
        new = pd.read_csv(C.TABLE_DIR / f"seeds_nfunsw_catdrop_{sc}.csv").set_index("seed")
        old = pd.read_csv(BACKUP / f"seeds_nfunsw_catdrop_{sc}.csv").set_index("seed")
        eps = [n_epochs(C.TABLE_DIR / f"history_nfunsw_catdrop_{sc}_m4{SEED_SUFFIX[s]}.csv")
               for s in (1, 2, 3)]
        print(f"  {sc:17s} epochs s1-s3 {eps}")
        for s in (1, 2, 3):
            a, b = old.loc[s, METRIC], new.loc[s, METRIC]
            print(f"    seed {s}: manuscript {a:.4f}  retrained {b:.4f}  diff {100 * (b - a):+.2f} pp")
        print(f"    mean 1-3: manuscript {old[METRIC].mean():.4f}  "
              f"retrained {new.loc[[1, 2, 3], METRIC].mean():.4f}")
        ref = (B_TAB / f"seeds_nfunsw_catdrop_{sc}.csv").stat().st_mtime
        stale = [s for s in (1, 2, 3) if (C.TABLE_DIR / f"history_nfunsw_catdrop_{sc}_m4"
                                           f"{SEED_SUFFIX[s]}.csv").stat().st_mtime < ref]
        if stale:
            raise SystemExit(f"[stop] {sc}: seeds {stale} were not retrained after isolate")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["isolate", "compare"], required=True)
    a = ap.parse_args()
    isolate() if a.step == "isolate" else compare()


if __name__ == "__main__":
    main()
