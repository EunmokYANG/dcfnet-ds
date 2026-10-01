"""1d 메커니즘 측정 - 저해상도 피처를 모델이 실제로 덜 쓰는가.

IEEE Access Access-2026-40899 심사 대응 (R4 Comment 2).

원고의 해석은 "좁은 대역에 갇힌 변동은 이를 보상하기 위해 더 큰 가중치를
필요로 하고 더 약한 gradient 를 낳는다" 는 최적화 기하 논리다. 이를
반증 가능한 사전 예측으로 바꾸어 측정한다.

    예측: central-range occupancy 가 낮은 피처일수록
          min-max 하에서 로짓 기여도와 제거 시 성능 손실이 모두 작다.
          signed-log 하에서는 둘 다 회복된다.

측정 지표 (재학습 불필요, 저장된 체크포인트만 사용)
  deg1_contrib  1차 로짓 기여도 |V1[j,c] * x~_j|      로짓 단위
  grad_x_input  전차수 기여도  |dlogit_c/dx_j * x_j|  로짓 단위
  perm_drop     피처 j 셔플 시 macro PR-AUC 감소       성능 단위
  w_norm        ||V1[j,:]||_2                          가중치 단위 (보조)

앞의 셋은 로짓 또는 성능 단위이므로 스케일러를 가로질러 비교할 수 있다.
w_norm 은 입력 단위에 의존하므로 보조 자료로만 제시하며, 논문에는 반드시
"weight magnitude alone is not interpreted as feature importance" 를
함께 적는다. 스케일러가 바뀌면 |w| 는 단위 환산만으로도 변하며, 이는
본 논문의 아핀 불변성 논지와 정면으로 충돌하기 때문이다.

실행:
  python -m src.p1_mechanism_v1 --dataset ciciot2023 --tag full_scalerminmax
  python -m src.p1_mechanism_v1 --dataset nfunsw --tag full_scalersignedlog --seed 2
  python -m src.p1_mechanism_v1 --dataset ciciot2023 --tag full_scalernone --skip-perm
"""

import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from tensorflow.keras import models

from src import config as C
from src.model import CUSTOM_OBJECTS

IND = ["deg1_contrib", "grad_x_input", "perm_drop", "w_norm"]


def model_path(stem, variant, seed):
    """multiseed.py 와 동일한 명명 규칙."""
    suffix = "" if seed == C.SEED else f"_s{seed}"
    return C.MODEL_DIR / f"{stem}_{variant}{suffix}{C.MODEL_EXT}"


def load_bundle(dataset, tag):
    stem = f"{dataset}_{tag}" if tag else dataset
    d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
    meta = json.loads(
        (C.PROCESSED_DIR / f"{stem}_meta.json").read_text("utf-8"))
    return d, meta, stem


def feature_diagnostics(meta, bundle, stem):
    """피처별 occupancy / rms / 고유값 수를 확보한다.

    bundle 은 npz 핸들이다. meta 나 캐시로 해결되면 x_train 을 아예 읽지
    않는다. NF-UNSW-NB15-v3 의 x_train 은 약 295MB 라 불필요한 로드를
    피하는 것이 반복 실행에서 유의미하다.

    진단 지표를 preprocess.py 에 추가하기 전에 만들어진 meta 에는 이 값이
    없다. 전처리를 다시 돌리면 npz 가 덮어써져 기존 체크포인트가 그 데이터로
    학습된 것인지 보장할 수 없으므로, 같은 정의로 x_train 에서 직접 계산한다.

        occupancy_j = (Q3_j - Q1_j) / (max_j - min_j)

    preprocess.py 의 iqr_ratio 와 동일한 식이며, 열 단위로 계산해
    float64 전체 사본을 만들지 않는다. 결과는 별도 캐시 파일에 남겨
    다음 실행에서 다시 계산하지 않는다.
    """
    need = ("feature_iqr_over_range", "feature_rms", "feature_n_unique")
    d = int(bundle["x_test"].shape[1])     # 피처 수만 알면 되므로 작은 쪽을 본다
    names = meta.get("feature_names") or [f"f{j}" for j in range(d)]
    if len(names) != d:                    # 길이가 어긋나면 자동 이름으로 대체
        names = [f"f{j}" for j in range(d)]

    if all(k in meta for k in need):
        return (names,
                np.asarray(meta["feature_iqr_over_range"], dtype="float64"),
                np.asarray(meta["feature_rms"], dtype="float64"),
                np.asarray(meta["feature_n_unique"], dtype="int64"))

    cache = C.PROCESSED_DIR / f"{stem}_featdiag.json"
    if cache.exists():
        c = json.loads(cache.read_text("utf-8"))
        print(f"  [진단 캐시] {cache.name} 재사용", flush=True)
        return (names,
                np.asarray(c["occupancy"], dtype="float64"),
                np.asarray(c["feature_rms"], dtype="float64"),
                np.asarray(c["n_unique"], dtype="int64"))

    print("  [진단 계산] meta 에 피처별 진단값이 없어 x_train 에서 "
          f"직접 계산합니다 ({d}개 피처)", flush=True)
    x_train = bundle["x_train"]            # 여기서만 실제로 읽는다
    occ = np.zeros(d, dtype="float64")
    frms = np.zeros(d, dtype="float64")
    nuni = np.zeros(d, dtype="int64")
    for j in range(d):
        col = x_train[:, j].astype("float64")
        frms[j] = np.sqrt(np.mean(col ** 2))
        q1, q3 = np.percentile(col, [25, 75])
        span = col.max() - col.min()
        occ[j] = (q3 - q1) / span if span > 0 else 0.0
        nuni[j] = len(np.unique(col))
    occ = np.nan_to_num(occ)

    cache.write_text(json.dumps({
        "source": "computed from x_train by p1_mechanism_v1",
        "occupancy": occ.tolist(),
        "feature_rms": frms.tolist(),
        "n_unique": nuni.tolist(),
    }), encoding="utf-8")
    print(f"  [진단 저장] {cache.name}", flush=True)
    return names, occ, frms, nuni


def stratified_sample(y, n_total, n_classes, rng, floor=50):
    """클래스가 하나도 빠지지 않도록 계층 표본을 뽑는다.

    macro PR-AUC 는 양성이 없는 클래스에서 정의되지 않으므로,
    희소 클래스에 최소 floor 개를 보장한다.
    """
    keep = []
    for c in range(n_classes):
        idx = np.flatnonzero(y == c)
        if len(idx) == 0:
            continue
        n = max(floor, int(round(len(idx) * n_total / len(y))))
        n = min(n, len(idx))
        keep.append(rng.choice(idx, size=n, replace=False))
    return np.sort(np.concatenate(keep))


def degree_scale(model, k=1):
    """DegreeScale 의 추론 시 스케일(편향 보정 포함)을 복원한다."""
    layer = model.get_layer(f"scale_deg{k}")
    # Keras 2 는 "scale_deg1/rms:0", Keras 3 는 "rms" 로 이름이 다르다.
    # 이름으로 먼저 찾고, 못 찾으면 build 순서(rms, step)로 되돌아간다.
    w = {}
    for v in layer.weights:
        key = v.name.split("/")[-1].split(":")[0]
        w[key] = np.asarray(v.numpy()).reshape(())
    if "rms" not in w or "step" not in w:
        vals = [np.asarray(v.numpy()).reshape(()) for v in layer.weights]
        w = {"rms": vals[0], "step": vals[1] if len(vals) > 1 else 0.0}
    s = float(w["rms"])
    step = float(w["step"])
    mom = float(getattr(layer, "momentum", 0.99))
    eps = float(getattr(layer, "eps", 1e-6))
    debias = 1.0 - mom ** max(step, 1.0)
    return s / max(debias, eps) + eps


def logit_model(model):
    """softmax 이전 로짓을 내보내는 부분 모델."""
    return models.Model(model.input, model.get_layer("logits").output)


def deg1_contribution(model, x, pred):
    """1차 로짓 기여도 |V1[j, pred] * x~_j| 의 표본 평균."""
    v1 = model.get_layer("logit_deg1").get_weights()[0]      # (d, C)
    scale = degree_scale(model, 1)
    xt = x / scale                                           # (N, d)
    sel = v1[:, pred].T                                      # (N, d)
    contrib = np.abs(xt * sel).mean(axis=0)
    return contrib, np.linalg.norm(v1, axis=1)


def grad_x_input(lm, x, pred, batch=4096):
    """전차수 Gradient x Input. 예측 클래스 로짓에 대한 기울기를 쓴다."""
    tot = np.zeros(x.shape[1], dtype="float64")
    for i in range(0, len(x), batch):
        xb = tf.convert_to_tensor(x[i:i + batch])
        idx = tf.convert_to_tensor(pred[i:i + batch].astype("int32"))
        with tf.GradientTape() as tape:
            tape.watch(xb)
            lg = lm(xb, training=False)
            # batch_dims 인자는 TF 버전마다 동작이 달라 one_hot 으로 고른다.
            oh = tf.one_hot(idx, tf.shape(lg)[-1], dtype=lg.dtype)
            sel = tf.reduce_sum(lg * oh, axis=1)
        g = tape.gradient(sel, xb)
        tot += np.abs(g.numpy() * xb.numpy()).sum(axis=0)
    return tot / len(x)


def macro_ap(model, x, y, n_classes, batch=8192):
    prob = model.predict(x, batch_size=batch, verbose=0)
    yb = label_binarize(y, classes=np.arange(n_classes))
    return float(average_precision_score(yb, prob, average="macro"))


def permutation_drop(model, x, y, n_classes, rng, repeats=1, batch=8192):
    """피처를 셔플해 macro PR-AUC 가 얼마나 떨어지는지 잰다.

    모델과 전처리를 블랙박스로 두고 성능으로만 재므로 단위 논쟁이
    발생하지 않는다. 네 지표 중 가장 방어적이다.
    """
    base = macro_ap(model, x, y, n_classes, batch)
    drop = np.zeros(x.shape[1], dtype="float64")
    for j in range(x.shape[1]):
        acc = 0.0
        for _ in range(repeats):
            xp = x.copy()
            xp[:, j] = xp[rng.permutation(len(xp)), j]
            acc += base - macro_ap(model, xp, y, n_classes, batch)
        drop[j] = acc / repeats
        print(f"    [perm] {j + 1}/{x.shape[1]}  drop={drop[j]:+.5f}",
              flush=True)
    return base, drop


def analyze(dataset, tag, variant, seed, n_sample, repeats, skip_perm):
    d, meta, stem = load_bundle(dataset, tag)
    path = model_path(stem, variant, seed)
    if not path.exists():
        print(f"[중단] 체크포인트 없음: {path.name}")
        return None

    n_classes = meta["n_classes"]
    names, occ, frms, nuni = feature_diagnostics(meta, d, stem)

    x_te, y_te = d["x_test"], d["y_test"]
    rng = np.random.default_rng(C.SEED)
    if n_sample and len(y_te) > n_sample:
        sel = stratified_sample(y_te, n_sample, n_classes, rng)
        x_te, y_te = x_te[sel], y_te[sel]
    print(f"[data] {stem} / {variant} / seed={seed}  "
          f"eval={x_te.shape}  classes={n_classes}", flush=True)

    model = models.load_model(path, custom_objects=CUSTOM_OBJECTS,
                              compile=False)
    prob = model.predict(x_te, batch_size=8192, verbose=0)
    pred = np.argmax(prob, axis=1)

    print("  [1/3] 1차 로짓 기여도", flush=True)
    c1, wnorm = deg1_contribution(model, x_te, pred)

    print("  [2/3] Gradient x Input", flush=True)
    gxi = grad_x_input(logit_model(model), x_te, pred)

    if skip_perm:
        print("  [3/3] permutation 생략", flush=True)
        base, drop = float("nan"), np.full(x_te.shape[1], np.nan)
    else:
        print(f"  [3/3] permutation ({x_te.shape[1]}개 피처 x {repeats}회)",
              flush=True)
        base, drop = permutation_drop(model, x_te, y_te, n_classes,
                                      rng, repeats)

    feat = pd.DataFrame({
        "dataset": dataset, "tag": tag, "scaler": meta.get("scaler"),
        "variant": variant, "seed": seed,
        "feature": names, "occupancy": occ, "feature_rms": frms,
        "n_unique": nuni,
        "deg1_contrib": c1, "grad_x_input": gxi,
        "perm_drop": drop, "w_norm": wnorm,
    })
    fp = C.TABLE_DIR / f"mech_features_{stem}_{variant}_s{seed}.csv"
    feat.to_csv(fp, index=False)

    # --- 예측 검증: occupancy 와 각 지표의 순위 상관
    cont = nuni > 2                       # 이진 피처는 occupancy 가 무의미
    row = {"dataset": dataset, "tag": tag, "scaler": meta.get("scaler"),
           "variant": variant, "seed": seed,
           "n_features": len(names), "n_continuous": int(cont.sum()),
           "occupancy_median": float(np.median(occ[cont])) if cont.any()
           else float("nan"),
           "base_macro_ap": base, "n_eval": int(len(y_te))}
    for k in IND:
        v = feat[k].to_numpy()
        ok = cont & np.isfinite(v) & np.isfinite(occ)
        if ok.sum() >= 4:
            r, p = spearmanr(occ[ok], v[ok])
            row[f"rho_{k}"] = round(float(r), 4)
            row[f"p_{k}"] = round(float(p), 4)
        else:
            row[f"rho_{k}"] = float("nan")
            row[f"p_{k}"] = float("nan")

    out = C.TABLE_DIR / "mech_summary.csv"
    df = pd.DataFrame([row])
    if out.exists():
        prev = pd.read_csv(out)
        key = ["dataset", "tag", "variant", "seed"]
        if all(k in prev.columns for k in key):
            df = pd.concat([prev, df], ignore_index=True)
            df = df.drop_duplicates(subset=key, keep="last")
            df = df.sort_values(key).reset_index(drop=True)
    df.to_csv(out, index=False)

    print(f"\n{'=' * 64}\n  occupancy 와의 순위 상관 ({stem})\n{'=' * 64}")
    for k in IND:
        tail = " (보조 - 단위 의존)" if k == "w_norm" else ""
        print(f"  {k:14s} rho = {row[f'rho_{k}']:+.4f}   "
              f"p = {row[f'p_{k}']:.4f}{tail}")
    print(f"\n  연속형 피처 {int(cont.sum())}/{len(names)}개로 계산")
    print(f"  [save] {fp.name}, mech_summary.csv")
    return feat, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--variant", default="m4")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--n-sample", type=int, default=100_000,
                    help="평가 표본 수. 0 이면 테스트 전체")
    ap.add_argument("--repeats", type=int, default=1,
                    help="permutation 반복 횟수")
    ap.add_argument("--skip-perm", action="store_true",
                    help="permutation 을 건너뛴다 (빠른 확인용)")
    a = ap.parse_args()
    analyze(a.dataset, a.tag, a.variant, a.seed, a.n_sample,
            a.repeats, a.skip_perm)


if __name__ == "__main__":
    main()