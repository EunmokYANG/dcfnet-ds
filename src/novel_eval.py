"""평가 집합을 '학습에서 본 피처 벡터'와 '처음 보는 벡터'로 나눠 재평가.

중복 행이 많은 데이터에서 성능이 암기에 의존하는지 확인한다.
학습은 하지 않고 저장된 모델을 다시 불러 예측만 수행한다.

마스크는 데이터셋당 한 번만 계산해 모든 스케일러에 재사용한다.
모든 스케일러가 피처별 단조 변환이고 분할 인덱스가 동일하므로
어떤 행이 중복인지는 스케일러와 무관하기 때문이다.
(float32 반올림으로 미세한 차이가 생길 수 있어 --recompute-mask 로 조건마다
 다시 계산할 수도 있다.)

실행:
  python -m src.novel_eval --dataset nfunsw --scalers minmax signedlog signedlog_minmax --seeds 3
  python -m src.novel_eval --dataset all --seeds 3
"""

import argparse
import json

import numpy as np
import pandas as pd
from tensorflow.keras import models

from src import config as C
from src.evaluate import metrics_of
from src.model import CUSTOM_OBJECTS

DEFAULT_SCALERS = ["minmax", "standard", "signedlog", "signedlog_minmax",
                   "none"]


def seen_mask(x_train, x_test):
    """x_test 각 행이 x_train 에 정확히 같은 행으로 존재하는가."""
    d = x_train.shape[1]
    dt = np.dtype((np.void, x_train.dtype.itemsize * d))
    tr = np.ascontiguousarray(x_train).view(dt).ravel()
    te = np.ascontiguousarray(x_test).view(dt).ravel()
    return np.isin(te, np.unique(tr))


def model_path(stem, variant, seed):
    """train.py 의 명명 규칙과 일치시킨다."""
    name = f"{stem}_{variant}" + (f"_s{seed}" if seed != C.SEED else "")
    return C.MODEL_DIR / f"{name}{C.MODEL_EXT}"


def run(dataset, scalers, n_seeds, variant, base, recompute):
    rows = []
    mask = None
    for sc in scalers:
        stem = f"{dataset}_{base}{sc}"
        npz = C.PROCESSED_DIR / f"{stem}.npz"
        if not npz.exists():
            print(f"  [건너뜀] {npz.name} 없음")
            continue
        d = np.load(npz, allow_pickle=True)
        x_tr, x_te, y_te = d["x_train"], d["x_test"], d["y_test"]
        meta = json.loads(
            (C.PROCESSED_DIR / f"{stem}_meta.json").read_text("utf-8"))
        n_classes = meta["n_classes"]

        if mask is None or recompute:
            mask = seen_mask(x_tr, x_te)
            print(f"  [mask] {stem}: seen {mask.sum():,} / {len(mask):,} "
                  f"({100 * mask.mean():.2f}%)", flush=True)

        seeds = list(range(C.SEED, C.SEED + n_seeds))
        for s in seeds:
            p = model_path(stem, variant, s)
            if not p.exists():
                print(f"  [건너뜀] {p.name} 없음")
                continue
            m = models.load_model(p, custom_objects=CUSTOM_OBJECTS,
                                  compile=False)
            prob = m.predict(x_te, batch_size=4096, verbose=0)
            pred = np.argmax(prob, axis=1)
            for subset, sel in [("all", slice(None)), ("seen", mask),
                                ("novel", ~mask)]:
                yy, pp, qq = y_te[sel], pred[sel], prob[sel]
                if len(yy) == 0:
                    continue
                r = metrics_of(yy, pp, qq, n_classes)
                r.update({"dataset": dataset, "scaler": sc, "seed": s,
                          "subset": subset, "n_rows": int(len(yy)),
                          "n_classes_present": int(len(np.unique(yy)))})
                rows.append(r)
            print(f"  [완료] {stem} seed={s}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS) + ["all"],
                    default="all")
    ap.add_argument("--scalers", nargs="+", default=DEFAULT_SCALERS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variant", default="m4")
    ap.add_argument("--base", default="full_scaler",
                    help="npz/model tag 접두 (기본: sweep 이 만든 full_scaler)")
    ap.add_argument("--recompute-mask", action="store_true",
                    help="스케일러마다 마스크를 다시 계산")
    a = ap.parse_args()

    names = list(C.DATASETS) if a.dataset == "all" else [a.dataset]
    rows = []
    for ds in names:
        print(f"\n===== {ds} =====", flush=True)
        rows += run(ds, a.scalers, a.seeds, a.variant, a.base,
                    a.recompute_mask)
    if not rows:
        raise SystemExit("[중단] 수집된 결과가 없습니다.")

    raw = pd.DataFrame(rows)
    out_raw = C.TABLE_DIR / "novel_eval_raw.csv"
    raw.to_csv(out_raw, index=False)

    key = ["dataset", "scaler", "subset"]
    agg = raw.groupby(key).agg(
        n_rows=("n_rows", "first"),
        n_seeds=("seed", "count"),
        **{f"{m}_{k}": (m, k)
           for m in ["PR-AUC(macro)", "F1-score", "MCC"]
           for k in ["mean", "std"]}
    ).reset_index()
    out = C.TABLE_DIR / "novel_eval.csv"
    agg.round(4).to_csv(out, index=False)

    print(f"\n{'=' * 70}\n  seen / novel 부분집합 평가\n{'=' * 70}")
    print(agg.round(4).to_string(index=False))
    print(f"\n  [save] {out.name}, {out_raw.name}")


if __name__ == "__main__":
    main()