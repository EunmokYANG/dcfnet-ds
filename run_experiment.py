"""전체 실험을 한 번에 실행하고 엑셀로 저장한다.

PyCharm 실행 구성 하나만 만들어 두면 전 단계가 순차 실행된다.
이미 끝난 단계는 자동으로 건너뛴다 (--force 로 재실행).

단계
  1. prepare     원본 -> raw CSV
  2. preprocess  raw -> npz (분할/스케일링)
  3. train       변형별 학습 (단일 시드)
  4. evaluate    성능 평가 + 클래스별 리포트
  5. baselines   logreg / rf / hgb
  6. degree      차수별 로짓 기여
  7. interactions 피처 조합 계수      <- 주 기여
  8. perturbation 섭동 검증           <- 인과성
  9. multiseed   다중 시드 반복
  10. compare    쌍별 통계 검정
  11. report     엑셀 통합

실행:
  python run_experiment.py                          전체 (기본 설정)
  python run_experiment.py --quick                  다중시드 생략, 시드 1개
  python run_experiment.py --dataset ciciot2023     한 데이터셋만
  python run_experiment.py --from interactions      중간부터
  python run_experiment.py --only report            엑셀만 재생성
"""

import argparse
import subprocess
import sys
import time

from src import config as C

STEPS = ["prepare", "preprocess", "train", "evaluate", "baselines",
         "degree", "interactions", "perturbation", "multiseed",
         "compare", "report"]

LABEL = {
    "prepare": "원본 -> raw CSV",
    "preprocess": "raw -> npz",
    "train": "모델 학습",
    "evaluate": "성능 평가",
    "baselines": "Baseline (logreg/rf/hgb)",
    "degree": "차수별 기여",
    "interactions": "피처 조합 계수",
    "perturbation": "섭동 검증",
    "multiseed": "다중 시드",
    "compare": "통계 검정",
    "report": "엑셀 통합",
}


def run(args, label, allow_fail=False):
    cmd = [sys.executable] + args
    print(f"\n{'=' * 68}\n  {label}\n  $ {' '.join(cmd[1:])}\n{'=' * 68}",
          flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    m, s = divmod(int(time.time() - t0), 60)
    if rc != 0:
        if allow_fail:
            print(f"\n[건너뜀] {label} 실패 (경과 {m}분 {s}초) — 계속 진행",
                  flush=True)
            return False
        sys.exit(f"\n[중단] {label} 실패 (경과 {m}분 {s}초)")
    print(f"\n[완료] {label}  ({m}분 {s}초)", flush=True)
    return True


def done(dataset, tag, step, variants):
    key = f"{dataset}_{tag}" if tag else dataset
    spec = C.DATASETS[dataset]
    stem = spec["raw_file"].replace(".csv", "")
    checks = {
        "prepare": (C.RAW_DIR / f"{stem}_full.csv").exists(),
        "preprocess": (C.PROCESSED_DIR / f"{key}.npz").exists(),
        "train": all((C.MODEL_DIR / f"{key}_{v}{C.MODEL_EXT}").exists()
                     for v in variants),
        "evaluate": (C.TABLE_DIR / f"metrics_{key}.csv").exists(),
        "baselines": (C.TABLE_DIR / f"metrics_baseline_{key}.csv").exists(),
        "degree": (C.TABLE_DIR / f"degree_contrib_{key}_m4.csv").exists(),
        "interactions": (C.TABLE_DIR / f"interactions_{key}_m4.csv").exists(),
        "perturbation": (C.TABLE_DIR / f"perturb_{key}_m4.csv").exists(),
        "multiseed": (C.TABLE_DIR / f"seeds_{key}.csv").exists(),
        "compare": (C.TABLE_DIR / f"compare_{key}.csv").exists(),
    }
    return checks.get(step, False)


def process(dataset, a, steps):
    tag = "full" if a.full else ""
    tag_arg = ["--tag", tag] if tag else []
    ds = ["--dataset", dataset]
    full = ["--full"] if a.full else []
    split = a.split
    if split == "temporal" and not C.DATASETS[dataset].get("time_col"):
        print(f"\n[안내] {dataset} 은 타임스탬프가 없어 무작위 분할을 씁니다.")
        split = "random"

    for step in steps:
        if step == "report":
            continue
        if done(dataset, tag, step, a.variants) and not a.force:
            print(f"[건너뜀] {step:<13} 이미 완료됨 (--force 로 재실행)")
            continue

        if step == "prepare":
            run(["-m", "src.prepare_data", *ds, "--max-per-class",
                 "0" if a.full else str(C.MAX_PER_CLASS)],
                f"{LABEL[step]} ({dataset})")
        elif step == "preprocess":
            run(["-m", "src.preprocess", *ds, *full,
                 "--scaler", a.scaler, "--split", split, *tag_arg],
                f"{LABEL[step]} ({dataset})")
        elif step == "train":
            for v in a.variants:
                run(["-m", "src.train", *ds, "--variant", v, *tag_arg],
                    f"학습 {dataset} / {v}")
        elif step == "evaluate":
            run(["-m", "src.evaluate", *ds, "--variants", *a.variants,
                 *tag_arg], f"{LABEL[step]} ({dataset})")
        elif step == "baselines":
            args = ["-m", "src.baselines", *ds, *tag_arg]
            if a.max_train:
                args += ["--max-train", str(a.max_train)]
            run(args, f"{LABEL[step]} ({dataset})", allow_fail=True)
        elif step == "degree":
            run(["-m", "src.degree_analysis", *ds, "--variant", "m4",
                 *tag_arg], f"{LABEL[step]} ({dataset})")
        elif step == "interactions":
            run(["-m", "src.interactions", *ds, "--variant", "m4",
                 "--top", str(a.top), *tag_arg],
                f"{LABEL[step]} ({dataset})")
        elif step == "perturbation":
            run(["-m", "src.perturbation", *ds, "--variant", "m4",
                 "--n-feat", str(a.n_feat),
                 "--n-control", str(a.n_control), *tag_arg],
                f"{LABEL[step]} ({dataset})", allow_fail=True)
        elif step == "multiseed":
            if a.seeds <= 1:
                print("[건너뜀] multiseed    --seeds 1 이므로 생략")
                continue
            run(["-m", "src.multiseed", *ds, "--variants", *a.variants,
                 "--seeds", str(a.seeds), *tag_arg],
                f"{LABEL[step]} ({dataset}, {a.seeds}시드)")
        elif step == "compare":
            run(["-m", "src.compare", *ds, *tag_arg],
                f"{LABEL[step]} ({dataset})", allow_fail=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS) + ["all"],
                    default="all")
    ap.add_argument("--full", action="store_true", default=True,
                    help="원본 전체 사용 (기본값)")
    ap.add_argument("--sampled", dest="full", action="store_false",
                    help="상한 샘플링본 사용")
    ap.add_argument("--scaler", default=C.SCALER,
                    choices=["minmax", "standard", "robust",
                             "signedlog", "none"])
    ap.add_argument("--split", default="temporal",
                    choices=["random", "temporal"],
                    help="시간 분할이 불가능한 데이터셋은 자동으로 random")
    ap.add_argument("--variants", nargs="+",
                    default=["m4", "m4g", "m4m", "nn_mlp"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--top", type=int, default=20, help="상위 조합 수")
    ap.add_argument("--n-feat", type=int, default=3, help="섭동 피처 수")
    ap.add_argument("--n-control", type=int, default=30, help="섭동 대조군 수")
    ap.add_argument("--max-train", type=int, default=0,
                    help="baseline 학습 표본 상한 (RF 가 느릴 때)")
    ap.add_argument("--from", dest="start", choices=STEPS, default="prepare")
    ap.add_argument("--only", choices=STEPS, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="다중 시드 생략, 대조군 축소 (빠른 확인용)")
    ap.add_argument("--out", default="dcfnet_ds_results.xlsx")
    a = ap.parse_args()

    if a.quick:
        a.seeds, a.n_control = 1, 20

    steps = [a.only] if a.only else STEPS[STEPS.index(a.start):]
    names = list(C.DATASETS) if a.dataset == "all" else [a.dataset]

    t0 = time.time()
    print(f"\n{'#' * 68}")
    print(f"  DCFNet-DS 전체 실험")
    print(f"  데이터셋 {names}   변형 {a.variants}   시드 {a.seeds}")
    print(f"  스케일러 {a.scaler}   분할 {a.split}   "
          f"{'원본 전체' if a.full else '샘플링본'}")
    print(f"{'#' * 68}")

    for n in names:
        print(f"\n\n{'#' * 68}\n#  {n}\n{'#' * 68}")
        process(n, a, steps)

    if "report" in steps or a.only == "report":
        run(["-m", "src.report", "--out", a.out], LABEL["report"])

    m, s = divmod(int(time.time() - t0), 60)
    h, m = divmod(m, 60)
    print(f"\n{'#' * 68}")
    print(f"  전체 완료  (경과 {h}시간 {m}분 {s}초)")
    print(f"  결과: outputs/{a.out}")
    print(f"{'#' * 68}\n")


if __name__ == "__main__":
    main()
