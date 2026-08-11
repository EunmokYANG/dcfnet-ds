"""피처 조합 -> 공격 유형 귀속 (본 논문의 주 기여).

M4 계열은 로짓이 차수별 기여의 합으로 정확히 분해되므로, 각 차수의
다항식 계수를 닫힌 형태로 계산할 수 있다.

    u_1 = x
    u_k = x * (W_k u~_{k-1}),   u~_k = u_k / s_k
    logit_c = sum_k V_k^(c) u~_k + b_c

편향이 없고 잔차도 없으므로 계수는 가중치 곱으로 전개된다.

    1차:  coef[c,i]     = V_1[c,i] / s_1
    2차:  coef[c,i,j]   = V_2[c,i] W_2[i,j] / (s_1 s_2)
    3차:  coef[c,i,j,k] = V_3[c,i] W_3[i,j] W_2[j,k] / (s_1 s_2 s_3)
    4차:  한 단계 더

SHAP interaction values 는 2차까지만 지원하지만, 여기서는 4차까지
정확한 계수를 얻는다. 조합 수가 지수적으로 늘어나므로 3차 이상은
상위 경로만 추적한다.

실행:
  python -m src.interactions --dataset ciciot2023 --tag full --variant m4
  python -m src.interactions --dataset ciciot2023 --tag full --top 30
"""

import argparse
import json

import numpy as np
import pandas as pd
from tensorflow.keras import models

from src import config as C
from src.model import CUSTOM_OBJECTS


def load_parts(dataset, tag, variant):
    stem = f"{dataset}_{tag}" if tag else dataset
    d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
    meta = json.loads(
        (C.PROCESSED_DIR / f"{stem}_meta.json").read_text("utf-8"))
    m = models.load_model(C.MODEL_DIR / f"{stem}_{variant}{C.MODEL_EXT}",
                          custom_objects=CUSTOM_OBJECTS, compile=False)
    return d, meta, m, stem


def extract_weights(model, max_degree):
    """V_k (readout), W_k (cross), s_k (스케일)을 뽑는다."""
    V, W, S = {}, {}, {}
    for k in range(1, max_degree + 1):
        # Dense(kernel) shape = (D, C) -> 전치해서 (C, D)
        V[k] = model.get_layer(f"logit_deg{k}").get_weights()[0].T
        S[k] = float(model.get_layer(f"scale_deg{k}").get_weights()[0])
        if k >= 2:
            W[k] = model.get_layer(f"cross_deg{k}_w").get_weights()[0].T
    return V, W, S


def degree1(V, S, feats, classes):
    """1차 계수: 개별 피처의 직접 기여."""
    coef = V[1] / S[1]
    rows = []
    for c, cn in enumerate(classes):
        for i in np.argsort(-np.abs(coef[c]))[:len(feats)]:
            rows.append({"Attack": cn, "degree": 1, "features": feats[i],
                         "coef": float(coef[c, i]),
                         "abs_coef": float(abs(coef[c, i]))})
    return pd.DataFrame(rows)


def degree2(V, W, S, feats, classes, top):
    """2차 계수: coef[c,i,j] = V_2[c,i] * W_2[i,j] / (s_1 s_2).

    x_i * x_j 형태이므로 (i,j) 와 (j,i) 를 합쳐서 보고한다.
    """
    scale = S[1] * S[2]
    rows = []
    for c, cn in enumerate(classes):
        M = (V[2][c][:, None] * W[2]) / scale        # (D, D)
        M = M + M.T                                  # 대칭화 (i!=j 는 두 경로)
        np.fill_diagonal(M, np.diag(M) / 2)          # x_i^2 는 한 번만
        idx = np.dstack(np.unravel_index(
            np.argsort(-np.abs(M), axis=None), M.shape))[0]
        seen, n = set(), 0
        for i, j in idx:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            name = (f"{feats[i]}^2" if i == j
                    else f"{feats[i]} x {feats[j]}")
            rows.append({"Attack": cn, "degree": 2, "features": name,
                         "coef": float(M[i, j]),
                         "abs_coef": float(abs(M[i, j]))})
            n += 1
            if n >= top:
                break
    return pd.DataFrame(rows)


def degree_high(V, W, S, feats, classes, k, top, beam=40):
    """3차 이상: 조합 수가 D^k 로 폭발하므로 빔 탐색으로 상위 경로만 추적.

    coef[c, i, j1, ..., j_{k-1}] = V_k[c,i] * W_k[i,j1] * ... * W_2[j_{k-2}, j_{k-1}]
    """
    scale = np.prod([S[t] for t in range(1, k + 1)])
    D = len(feats)
    rows = []
    for c, cn in enumerate(classes):
        # 시작: (계수, 경로, 현재 인덱스)
        v = V[k][c] / scale
        cand = sorted(range(D), key=lambda i: -abs(v[i]))[:beam]
        paths = [(v[i], [i], i) for i in cand]
        for step in range(k, 1, -1):
            nxt = []
            Wm = W[step]
            for val, path, cur in paths:
                row = val * Wm[cur]
                for j in np.argsort(-np.abs(row))[:beam]:
                    nxt.append((float(row[j]), path + [int(j)], int(j)))
            nxt.sort(key=lambda t: -abs(t[0]))
            paths = nxt[:beam * 2]
        seen, n = set(), 0
        for val, path, _ in sorted(paths, key=lambda t: -abs(t[0])):
            key = tuple(sorted(path))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"Attack": cn, "degree": k,
                         "features": " x ".join(feats[t] for t in key),
                         "coef": val, "abs_coef": abs(val)})
            n += 1
            if n >= top:
                break
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--variant", default="m4")
    ap.add_argument("--top", type=int, default=20,
                    help="클래스·차수별 상위 조합 수")
    ap.add_argument("--max-degree", type=int, default=None)
    a = ap.parse_args()

    d, meta, model, stem = load_parts(a.dataset, a.tag, a.variant)
    feats = [str(f) for f in d["feature_names"]]
    classes = C.DATASETS[a.dataset]["classes"]
    K = a.max_degree or C.MAX_DEGREE

    V, W, S = extract_weights(model, K)
    print(f"[weights] V_k {V[1].shape}  W_2 {W[2].shape}")
    print("[scale]   " + "  ".join(f"s{k}={S[k]:.4g}" for k in range(1, K + 1)))

    parts = [degree1(V, S, feats, classes).groupby("Attack").head(a.top),
             degree2(V, W, S, feats, classes, a.top)]
    for k in range(3, K + 1):
        parts.append(degree_high(V, W, S, feats, classes, k, a.top))
    df = pd.concat(parts, ignore_index=True)

    out = C.TABLE_DIR / f"interactions_{stem}_{a.variant}.csv"
    df.to_csv(out, index=False)

    print(f"\n{'=' * 76}\n  클래스별 최상위 조합 (차수별 1위)\n{'=' * 76}")
    print(f"  {'Attack':<16}{'deg':>4}  {'조합':<44}{'계수':>10}")
    print("  " + "-" * 74)
    for cn in classes:
        sub = df[df.Attack == cn]
        for k in range(1, K + 1):
            r = sub[sub.degree == k].nlargest(1, "abs_coef")
            if len(r):
                r = r.iloc[0]
                nm = r["features"][:42]
                print(f"  {cn if k == 1 else '':<16}{k:>4}  {nm:<44}"
                      f"{r['coef']:>10.4f}")
        print()

    print(f"[save] {out.name}   총 {len(df):,}행")


if __name__ == "__main__":
    main()
