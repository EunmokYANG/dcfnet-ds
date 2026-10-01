"""1단계 Baseline.

표 형식 데이터에서 GBDT 가 신경망을 상회하는 경우가 흔하므로,
제안 모델의 위치를 먼저 확정해야 논문 프레이밍이 정해진다.

포함 모델
  logreg : Logistic Regression      1차 상호작용만 사용하는 하한선
  rf     : Random Forest            스케일 불변, 표 형식 표준
  hgb    : HistGradientBoosting     LightGBM 계열. 표 형식 최강 후보
  (신경망 MLP 는 src.train --variant nn_mlp 로 별도 학습)

실행:
  python -m src.baselines --dataset ciciot2023 --tag full
  python -m src.baselines --dataset nfunsw --tag full --models hgb rf
  python -m src.baselines --dataset nfunsw --tag full --max-train 400000
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config as C
from src.evaluate import metrics_of, per_class_report

# full training partition 으로 옮기면 400,000행 기준 상한(2000)으로는
# 부족할 수 있으므로 CLI 에서 올릴 수 있게 모듈 변수로 뺀다.
# solver 는 기존 lbfgs 를 그대로 유지한다 - solver 까지 바꾸면 이전
# Table IV 와의 차이가 scaler 때문인지 solver 때문인지 설명이 필요해진다.
#
# matched protocol 재실행의 기본 상한은 10000 으로 둔다. 인자를 빠뜨려
# 조건마다 상한이 달라지면 "스케일러 때문에 낮은 값" 과 "덜 돌아서 낮은
# 값" 이 섞여 비교가 성립하지 않는다. 조건별 실제 반복수는 n_iter 로
# 보고하므로 상한을 넉넉히 두는 데 따르는 손해는 없다.
LOGREG_MAX_ITER = 10000

# 수렴 기준을 가진 최적화기를 쓰는 모델만 수렴 여부를 판정한다.
# 부스팅의 반복 횟수는 수렴 지표가 아니므로 여기에 넣지 않는다.
ITERATIVE_SOLVERS = {"logreg"}

MODELS = {
    # max_iter=200 에서는 다섯 스케일러 모두 lbfgs 가 수렴하지 않았다.
    # 그 상태로 비교하면 스케일러 간 차이가 모델의 표현력이 아니라
    # 최적화 문제의 조건수를 재는 것이 되므로 상한을 올린다.
    # n_jobs 는 scikit-learn 1.8부터 무효이므로 제거한다.
    "logreg": lambda seed: LogisticRegression(
        max_iter=LOGREG_MAX_ITER, class_weight="balanced", random_state=seed),
    "rf": lambda seed: RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed),
    "hgb": lambda seed: HistGradientBoostingClassifier(
        max_iter=200, early_stopping=True, validation_fraction=0.1,
        class_weight="balanced", random_state=seed),
}

DESC = {
    "logreg": "Logistic Regression (linear, 1st order only)",
    "rf": "Random Forest (scale-invariant)",
    "hgb": "HistGradientBoosting (tabular SOTA candidate)",
}


def load(dataset, tag):
    stem = f"{dataset}_{tag}" if tag else dataset
    d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
    meta = json.loads(
        (C.PROCESSED_DIR / f"{stem}_meta.json").read_text("utf-8"))
    return d, meta, stem


def run(dataset, tag, model_names, max_train, seed):
    d, meta, stem = load(dataset, tag)
    names = C.DATASETS[dataset]["classes"]
    n_classes = meta["n_classes"]

    x_tr, y_tr = d["x_train"], d["y_train"]
    x_te, y_te = d["x_test"], d["y_test"]

    if max_train and len(y_tr) > max_train:
        rng = np.random.default_rng(seed)
        # 계층 축소: 클래스 비율을 유지한 채 표본 수만 줄인다
        keep = []
        for c in np.unique(y_tr):
            idx = np.flatnonzero(y_tr == c)
            n = max(1, int(round(len(idx) * max_train / len(y_tr))))
            keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
        sel = np.sort(np.concatenate(keep))
        x_tr, y_tr = x_tr[sel], y_tr[sel]
        print(f"[축소] train {len(sel):,}행으로 계층 축소")

    print(f"[data] train={x_tr.shape}  test={x_te.shape}  "
          f"classes={n_classes}")

    rows = []
    for name in model_names:
        print(f"\n{'=' * 58}\n  {name} : {DESC[name]}\n{'=' * 58}", flush=True)
        clf = MODELS[name](seed)
        t0 = time.time()
        clf.fit(x_tr, y_tr)
        fit_s = time.time() - t0

        # 반복 상한에 걸린 조건은 성능을 하한으로만 읽어야 한다.
        # 수렴 실패로 인한 저하와 스케일러로 인한 저하를 구분하기 위해
        # n_iter_ 와 수렴 여부를 지표와 함께 기록한다.
        #
        # 수렴 판정은 수렴 기준을 가진 최적화기에만 적용한다.
        # HistGradientBoosting 의 n_iter_ 는 부스팅 라운드 수이므로
        # max_iter 도달은 조기 종료 미발동일 뿐 최적화 실패가 아니다.
        # 이를 '미수렴'으로 표기하면 트리 대조군 성능이 하한으로 읽혀
        # 대조 논리 자체가 약해진다.
        n_it, cap, conv = None, None, None
        if name in ITERATIVE_SOLVERS and hasattr(clf, "n_iter_"):
            n_it = int(np.max(np.atleast_1d(clf.n_iter_)))
            cap = int(getattr(clf, "max_iter", 0)) or None
            conv = bool(cap is None or n_it < cap)
        elif hasattr(clf, "n_iter_"):
            # 참고용으로 라운드 수만 남기고 수렴 판정은 하지 않는다.
            n_it = int(np.max(np.atleast_1d(clf.n_iter_)))
            cap = int(getattr(clf, "max_iter", 0)) or None

        prob = clf.predict_proba(x_te)
        pred = np.argmax(prob, axis=1)
        m = metrics_of(y_te, pred, prob, n_classes)
        m.update({"Variant": name, "Desc": DESC[name], "seed": seed,
                  "n_train": int(len(y_tr)), "fit_sec": round(fit_s, 1),
                  "n_iter": n_it, "max_iter": cap, "converged": conv})
        rows.append(m)
        msg = (f"  학습 {fit_s:.1f}초  PR-AUC(macro)={m['PR-AUC(macro)']:.4f}  "
               f"F1={m['F1-score']:.4f}  MCC={m['MCC']:.4f}")
        if n_it is not None and conv is None:
            # 수렴 판정 대상이 아닌 모델(부스팅 등)은 라운드 수만 표시한다.
            # conv 가 None 이면 삼항식에서 거짓으로 떨어져 '상한도달' 로
            # 잘못 찍히므로 분기를 먼저 둔다.
            msg += f"  n_iter={n_it}/{cap}"
        elif n_it is not None:
            msg += (f"  n_iter={n_it}/{cap} "
                    f"{'수렴' if conv else '상한도달(하한으로 보고)'}")
        print(msg)

        pc = per_class_report(y_te, prob, names)
        pc.insert(0, "Variant", name)
        pc.to_csv(C.TABLE_DIR / f"perclass_{stem}_{name}_s{seed}.csv",
                  index=False)
        print(pc.round(4).to_string(index=False))

    cols = ["Variant", "Desc", "seed", "n_train", "Accuracy", "Precision",
            "Recall", "F1-score", "MCC", "PR-AUC(macro)", "PR-AUC(micro)",
            "fit_sec", "n_iter", "max_iter", "converged"]
    df = pd.DataFrame(rows)[cols]
    # 시드별로 따로 남긴다. 기존 단일 시드 결과 파일은 덮어쓰지 않으므로
    # 이전 Table IV 값과 대조할 수 있다.
    out = C.TABLE_DIR / f"metrics_baseline_{stem}_s{seed}.csv"
    saved = df
    if out.exists():
        # 모델을 나눠 실행해도 앞 모델의 행이 사라지지 않도록 병합해 저장한다.
        # logreg 를 단독 실행하면서 hgb 의 시드 1 행을 지운 사고가 있었다.
        # 반환값은 이번 호출의 행만 유지해 run_multi 의 집계를 흐리지 않는다.
        prev = pd.read_csv(out)
        if "Variant" in prev.columns:
            saved = pd.concat([prev, df], ignore_index=True)
            saved = saved.drop_duplicates(subset=["Variant"], keep="last")
            saved = saved.sort_values("Variant").reset_index(drop=True)
    saved.to_csv(out, index=False)

    print(f"\n{'=' * 58}\n  Baseline 요약 ({stem})\n{'=' * 58}")
    print(df.round(4).to_string(index=False))

    # 제안 모델과 한 표에 합쳐 비교
    prop = C.TABLE_DIR / f"metrics_{stem}.csv"
    if prop.exists():
        both = pd.concat([pd.read_csv(prop), df], ignore_index=True)
        both = both.sort_values("PR-AUC(macro)", ascending=False)
        both.to_csv(C.TABLE_DIR / f"metrics_all_{stem}_s{seed}.csv",
                    index=False)
        print(f"\n{'=' * 58}\n  전체 비교 (PR-AUC macro 내림차순)\n{'=' * 58}")
        print(both[["Variant", "Accuracy", "F1-score", "MCC",
                    "PR-AUC(macro)", "PR-AUC(micro)"]]
              .round(4).to_string(index=False))
    print(f"\n[save] {out.name}")
    return df

def run_multi(dataset, tag, model_names, max_train, seeds, seed_start):
    """시드를 바꿔 run() 을 반복하고 평균 +- 표준편차로 모은다.

    제안 모델과 동일한 다중 시드 프로토콜을 베이스라인에도 적용해,
    보고된 스케일러 span 을 같은 noise floor 위에서 비교할 수 있게 한다.
    수렴하지 않은 조건 수를 함께 집계해 하한 보고 대상을 표시한다.
    """
    stem = f"{dataset}_{tag}" if tag else dataset
    parts = []
    for s in range(seed_start, seed_start + seeds):
        print(f"\n{'#' * 58}\n  {stem}  seed = {s}\n{'#' * 58}", flush=True)
        parts.append(run(dataset, tag, model_names, max_train, s))

    raw = pd.concat(parts, ignore_index=True)
    out = C.TABLE_DIR / f"seeds_baseline_{stem}.csv"
    if out.exists():
        # 모델을 나눠 실행해도 먼저 측정한 행이 사라지지 않도록 병합한다.
        prev = pd.read_csv(out)
        if {"Variant", "seed"} <= set(prev.columns):
            raw = pd.concat([prev, raw], ignore_index=True)
            raw = raw.drop_duplicates(subset=["Variant", "seed"], keep="last")
            raw = raw.sort_values(["Variant", "seed"]).reset_index(drop=True)
    raw.to_csv(out, index=False)

    keys = ["Accuracy", "Precision", "Recall", "F1-score", "MCC",
            "PR-AUC(macro)", "PR-AUC(micro)"]
    g = raw.groupby("Variant")[keys]
    summ = g.mean().round(4).add_suffix("_mean").join(
        g.std().round(4).add_suffix("_std"))
    summ["n_seeds"] = g.size()
    if "converged" in raw.columns:
        summ["n_not_converged"] = raw.groupby("Variant")["converged"].apply(
            lambda c: int((c.astype("object") == False).sum()))
    summ = summ.reset_index().sort_values("PR-AUC(macro)_mean",
                                          ascending=False)
    summ.to_csv(C.TABLE_DIR / f"summary_baseline_{stem}.csv", index=False)

    print(f"\n{'=' * 70}\n  베이스라인 다중 시드 요약 ({stem})\n{'=' * 70}")
    for _, r in summ.iterrows():
        line = (f"  {r['Variant']:8s}  PR-AUC(macro) "
                f"{r['PR-AUC(macro)_mean']:.4f} +- "
                f"{r['PR-AUC(macro)_std']:.4f}   n={int(r['n_seeds'])}")
        if "n_not_converged" in summ.columns and r["n_not_converged"]:
            line += f"   [미수렴 {int(r['n_not_converged'])}건 - 하한]"
        print(line)
    print(f"\n  [save] {out.name}, summary_baseline_{stem}.csv")
    return raw, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS))
    ap.add_argument("--max-train", type=int, default=0,
                    help="학습 표본 상한. RF 가 느리면 400000 등으로 지정")
    ap.add_argument("--seed", type=int, default=C.SEED)
    # 제안 모델과 동일한 프로토콜로 맞추기 위한 인자
    ap.add_argument("--seeds", type=int, default=1,
                    help="시드 개수. 2 이상이면 다중 시드로 집계한다")
    ap.add_argument("--seed-start", type=int, default=None,
                    help="시작 시드. 기본값은 --seed 값")
    ap.add_argument("--logreg-max-iter", type=int, default=None,
                    help="logreg 의 lbfgs 반복 상한. solver 는 바꾸지 않는다")
    a = ap.parse_args()

    global LOGREG_MAX_ITER
    if a.logreg_max_iter:
        LOGREG_MAX_ITER = a.logreg_max_iter

    start = a.seed_start if a.seed_start is not None else a.seed
    if a.seeds > 1:
        run_multi(a.dataset, a.tag, a.models, a.max_train, a.seeds, start)
    else:
        run(a.dataset, a.tag, a.models, a.max_train, start)


if __name__ == "__main__":
    main()
