"""DCFNet-DS 파이프라인 관리 스크립트.

현재 어느 단계까지 끝났는지 파악하고, 남은 단계만 실행한다.

사용 예)
  python main.py --status                       현재 상태만 확인
  python main.py --dataset ciciot2023 --full    원본 전체로 전 단계 실행
  python main.py --dataset nfunsw --full --split temporal
  python main.py --dataset ciciot2023 --full --only preprocess
  python main.py --dataset ciciot2023 --full --from train
  python main.py --dataset ciciot2023 --full --force   완료된 단계도 재실행
"""

import argparse
import subprocess
import sys
import time

from src import config as C

STAGES = ["prepare", "preprocess", "train", "evaluate", "degree", "branch"]

LABEL = {
    "prepare": "원본 -> raw CSV",
    "preprocess": "raw -> npz (분할/스케일링)",
    "train": "모델 학습",
    "evaluate": "성능 평가",
    "degree": "차수 게이트 분석",
    "branch": "분기/차수 차단 분석",
}


# ---------------------------------------------------------------- 상태 조회
def paths(dataset, full, tag):
    spec = C.DATASETS[dataset]
    stem = spec["raw_file"].replace(".csv", "")
    raw = C.RAW_DIR / (f"{stem}_full.csv" if full else f"{stem}.csv")
    key = f"{dataset}_{tag}" if tag else dataset
    return {
        "source": C.SOURCE_DIR / spec["raw_file"],
        "raw": raw,
        "npz": C.PROCESSED_DIR / f"{key}.npz",
        "meta": C.PROCESSED_DIR / f"{key}_meta.json",
        "key": key,
    }


def mb(p):
    return f"{p.stat().st_size / 1024 / 1024:,.0f} MB" if p.exists() else "-"


def state(dataset, full, tag, variants):
    p = paths(dataset, full, tag)
    models = {v: C.MODEL_DIR / f"{p['key']}_{v}{C.MODEL_EXT}" for v in variants}
    return {
        "prepare": p["raw"].exists(),
        "preprocess": p["npz"].exists() and p["meta"].exists(),
        "train": all(m.exists() for m in models.values()),
        "evaluate": (C.TABLE_DIR / f"metrics_{p['key']}.csv").exists(),
        "degree": any((C.TABLE_DIR / f"{pre}_{p['key']}_{v}.csv").exists()
                      for v in variants
                      for pre in ("degree_contrib", "degree_alpha")),
        "branch": any((C.TABLE_DIR / f"occl_branch_{p['key']}_{v}.csv").exists()
                      for v in variants)
                  or all(v.startswith("m4") for v in variants),
        "_paths": p,
        "_models": models,
    }


def show_status(dataset, full, tag, variants):
    st = state(dataset, full, tag, variants)
    p, models = st["_paths"], st["_models"]
    print(f"\n{'=' * 62}")
    print(f"  {dataset}   (full={full}, tag={tag or '-'})")
    print("=" * 62)
    print(f"  source : {'O' if p['source'].exists() else 'X'}  "
          f"{p['source'].name:<28} {mb(p['source'])}")
    print(f"  raw    : {'O' if p['raw'].exists() else 'X'}  "
          f"{p['raw'].name:<28} {mb(p['raw'])}")
    print(f"  npz    : {'O' if p['npz'].exists() else 'X'}  "
          f"{p['npz'].name:<28} {mb(p['npz'])}")
    print("  models :")
    for v, m in models.items():
        print(f"           {'O' if m.exists() else 'X'}  {v:<4} {mb(m)}")
    print("  " + "-" * 58)
    for s in STAGES:
        print(f"  [{'완료' if st[s] else '  ? '}] {s:<11} {LABEL[s]}")
    print()
    return st


# ---------------------------------------------------------------- 실행
def run(args_list, label):
    cmd = [sys.executable] + args_list
    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd[1:])}")
    print("=" * 62, flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    el = time.time() - t0
    m, s = divmod(int(el), 60)
    if rc != 0:
        sys.exit(f"\n[중단] {label} 실패 (경과 {m}분 {s}초)")
    print(f"\n[완료] {label}  (경과 {m}분 {s}초)", flush=True)


def build_cmds(dataset, a, tag):
    key = f"{dataset}_{tag}" if tag else dataset
    tag_arg = ["--tag", tag] if tag else []
    cmds = {
        "prepare": ["-m", "src.prepare_data", "--dataset", dataset,
                    "--max-per-class", "0" if a.full else str(C.MAX_PER_CLASS)],
        "preprocess": ["-m", "src.preprocess", "--dataset", dataset,
                       *(["--full"] if a.full else []),
                       "--scaler", a.scaler, "--split", a.split, *tag_arg],
        "evaluate": ["-m", "src.evaluate", "--dataset", dataset,
                     "--variants", *a.variants, *tag_arg],
        "degree": ["-m", "src.degree_analysis", "--dataset", dataset,
                   "--variant", a.variants[-1], *tag_arg],
        "branch": ["-m", "src.branch_analysis", "--dataset", dataset,
                   "--variant", a.variants[-1], *tag_arg],
    }
    return cmds, key


def process(dataset, a):
    tag = a.tag if a.tag is not None else ("full" if a.full else "")
    st = show_status(dataset, a.full, tag, a.variants)
    if a.status:
        return

    todo = STAGES[STAGES.index(a.start):]
    if a.only:
        todo = [a.only]

    cmds, key = build_cmds(dataset, a, tag)
    tag_arg = ["--tag", tag] if tag else []

    for s in todo:
        if st[s] and not a.force:
            print(f"[건너뜀] {s:<11} 이미 완료됨 (--force 로 재실행)")
            continue
        if s == "train":
            for v in a.variants:
                m = st["_models"][v]
                if m.exists() and not a.force:
                    print(f"[건너뜀] train {v} 이미 완료됨")
                    continue
                run(["-m", "src.train", "--dataset", dataset,
                     "--variant", v, *tag_arg], f"학습 {key} / {v}")
        else:
            run(cmds[s], f"{LABEL[s]} ({key})")

    print(f"\n{'=' * 62}\n  {dataset} 완료. outputs/ 를 확인하세요.\n{'=' * 62}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS) + ["all"],
                    default="all")
    ap.add_argument("--full", action="store_true", help="원본 전체 사용")
    ap.add_argument("--scaler", default=C.SCALER,
                    choices=["minmax", "standard", "robust",
                             "signedlog", "none"])
    ap.add_argument("--split", default=C.SPLIT,
                    choices=["random", "temporal"])
    ap.add_argument("--tag", default=None,
                    help="출력 접미사. 미지정 시 full 이면 'full'")
    ap.add_argument("--variants", nargs="+", default=["m4"])
    ap.add_argument("--from", dest="start", choices=STAGES, default="prepare")
    ap.add_argument("--only", choices=STAGES, default=None)
    ap.add_argument("--force", action="store_true", help="완료 단계도 재실행")
    ap.add_argument("--status", action="store_true", help="상태만 출력")
    a = ap.parse_args()

    names = list(C.DATASETS) if a.dataset == "all" else [a.dataset]
    for n in names:
        if a.split == "temporal" and not C.DATASETS[n].get("time_col"):
            print(f"\n[건너뜀] {n} 은 타임스탬프가 없어 시간 분할 불가")
            continue
        process(n, a)


if __name__ == "__main__":
    main()
