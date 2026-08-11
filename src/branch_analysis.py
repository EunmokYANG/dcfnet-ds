"""분기/차수 경로 차단(occlusion) 분석.

차수 게이트 alpha_k 는 학습된 가중치일 뿐이므로, 그 값이 실제로 판단에
기여하는지는 별도로 측정해야 한다. 이 스크립트는 특정 경로를 0 으로 만든 뒤
클래스별 정확도가 얼마나 떨어지는지를 잰다.

측정 항목
  1. 분기 차단 : MLP 분기 vs 교차 분기 중 어느 쪽이 판단을 주도하는가
  2. 차수 차단 : u_k 를 하나씩 끊었을 때 어느 클래스가 무너지는가
  3. 정합성   : alpha_k 가 지목한 차수와 차단 시 손실이 큰 차수가 일치하는가

실행:  python -m src.branch_analysis --dataset ciciot2023 --variant m3
"""

import argparse
import json

import numpy as np
import pandas as pd
from tensorflow.keras import layers, models

from src import config as C
from src.model import CUSTOM_OBJECTS


def split_model(model):
    """모델을 (특징 추출부, 융합 헤드) 로 분리한다.

    반환:
      feat  : inputs -> [mlp_out, cross_out]
      head  : concat 결과 -> 확률
      names : (mlp 차원, cross 차원)
    """
    concat = None
    ci = None
    for i, lyr in enumerate(model.layers):
        if isinstance(lyr, layers.Concatenate):
            concat, ci = lyr, i
            break
    if concat is None:
        raise RuntimeError("Concatenate 레이어를 찾지 못했습니다.")

    mlp_t, cross_t = concat.input
    feat = models.Model(model.input, [mlp_t, cross_t], name="feat")

    d_mlp = int(mlp_t.shape[-1])
    d_cross = int(cross_t.shape[-1])

    head_in = layers.Input(shape=(d_mlp + d_cross,))
    h = head_in
    for lyr in model.layers[ci + 1:]:
        h = lyr(h)
    head = models.Model(head_in, h, name="head")
    return feat, head, d_mlp, d_cross


def degree_parts(model, x, max_degree):
    """alpha 와 u_1..u_K 를 얻는다. u_1 은 입력 자신이다."""
    outs = [model.get_layer("degree_alpha").output]
    for k in range(2, max_degree + 1):
        outs.append(model.get_layer(f"cross_deg{k}").output)
    sub = models.Model(model.input, outs)
    res = sub.predict(x, batch_size=1024, verbose=0)
    alpha, us = res[0], [x] + list(res[1:])
    return alpha, us


def per_class_acc(y_true, prob, n_classes):
    pred = np.argmax(prob, axis=1)
    return np.array([
        (pred[y_true == c] == c).mean() if (y_true == c).sum() else np.nan
        for c in range(n_classes)
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--variant", default="m3")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    stem = f"{a.dataset}_{a.tag}" if a.tag else a.dataset
    d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
    meta = json.loads(
        (C.PROCESSED_DIR / f"{stem}_meta.json").read_text("utf-8")
    )
    names = C.DATASETS[a.dataset]["classes"]
    n_classes = meta["n_classes"]
    x, y = d["x_test"], d["y_test"]

    model = models.load_model(
        C.MODEL_DIR / f"{stem}_{a.variant}{C.MODEL_EXT}",
        custom_objects=CUSTOM_OBJECTS, compile=False,
    )

    if any(l.name == "logit_deg1" for l in model.layers):
        print("\n[안내] 선형 readout 구조(M4 계열)에서는 차단 분석이 "
              "불필요합니다.")
        print("       각 차수의 로짓 기여 phi_k = V_k u~_k 가 항등식으로 "
              "정확히 계산되므로,")
        print("       경로를 끊어 추정할 필요가 없습니다. "
              "src.degree_analysis 를 사용하세요.")
        return

    feat, head, d_mlp, d_cross = split_model(model)
    print(f"[구조] MLP {d_mlp}차원 / 교차 {d_cross}차원 "
          f"(교차 비중 {100 * d_cross / (d_mlp + d_cross):.1f}%)")

    mlp_out, cross_out = feat.predict(x, batch_size=1024, verbose=0)
    zero_m, zero_c = np.zeros_like(mlp_out), np.zeros_like(cross_out)

    base = head.predict(np.concatenate([mlp_out, cross_out], 1),
                        batch_size=1024, verbose=0)
    ref = model.predict(x, batch_size=1024, verbose=0)
    gap = float(np.max(np.abs(base - ref)))
    print(f"[검증] 분해 재구성 오차 {gap:.2e} "
          f"({'정상' if gap < 1e-4 else '경고: 분해 실패'})")

    acc0 = per_class_acc(y, base, n_classes)

    # ---------------- 1. 분기 차단
    no_cross = head.predict(np.concatenate([mlp_out, zero_c], 1),
                            batch_size=1024, verbose=0)
    no_mlp = head.predict(np.concatenate([zero_m, cross_out], 1),
                          batch_size=1024, verbose=0)

    rows = []
    for c, nm in enumerate(names):
        rows.append({
            "Attack": nm, "N": int((y == c).sum()),
            "acc_full": acc0[c],
            "acc_no_cross": per_class_acc(y, no_cross, n_classes)[c],
            "acc_no_mlp": per_class_acc(y, no_mlp, n_classes)[c],
        })
    bdf = pd.DataFrame(rows)
    bdf["drop_cross"] = bdf.acc_full - bdf.acc_no_cross
    bdf["drop_mlp"] = bdf.acc_full - bdf.acc_no_mlp
    bdf.loc[len(bdf)] = {
        "Attack": "ALL", "N": len(y),
        "acc_full": (np.argmax(base, 1) == y).mean(),
        "acc_no_cross": (np.argmax(no_cross, 1) == y).mean(),
        "acc_no_mlp": (np.argmax(no_mlp, 1) == y).mean(),
        "drop_cross": np.nan, "drop_mlp": np.nan,
    }
    bdf.loc[len(bdf) - 1, "drop_cross"] = (
        bdf.loc[len(bdf) - 1, "acc_full"] - bdf.loc[len(bdf) - 1, "acc_no_cross"])
    bdf.loc[len(bdf) - 1, "drop_mlp"] = (
        bdf.loc[len(bdf) - 1, "acc_full"] - bdf.loc[len(bdf) - 1, "acc_no_mlp"])

    print("\n[1] 분기 차단 — 어느 분기가 판단을 주도하는가")
    print(bdf.round(4).to_string(index=False))
    bdf.to_csv(C.TABLE_DIR / f"occl_branch_{stem}_{a.variant}.csv", index=False)

    # ---------------- 2. 차수 차단
    if a.variant == "m0":
        print("\n[2] m0 는 차수 분리 구조가 없어 차수 차단을 건너뜁니다.")
        return

    K = C.MAX_DEGREE
    alpha, us = degree_parts(model, x, K)

    # ---------------- 2-0. 차수별 크기 진단
    norms = np.array([np.sqrt((u ** 2).mean()) for u in us])
    a_mean = alpha.mean(axis=0)
    eff = a_mean * norms
    ndf = pd.DataFrame({
        "degree": range(1, K + 1),
        "rms_u_k": norms,
        "alpha_mean": a_mean,
        "effective": eff,
        "effective_share": eff / eff.sum(),
    })
    print("\n[2-0] 차수별 크기 진단 — alpha 가 비교 가능한 값인가")
    print(ndf.round(5).to_string(index=False))
    print(f"      크기 비 rms(u1)/rms(u{K}) = {norms[0] / norms[-1]:.1f} 배")
    if norms.max() / norms.min() > 5:
        print("      -> u_k 크기가 크게 달라 alpha 단독 비교는 무의미합니다.")
        print("         effective_share 를 해석 지표로 써야 합니다.")
    ndf.to_csv(C.TABLE_DIR / f"degree_norm_{stem}_{a.variant}.csv", index=False)

    rows = []
    for c, nm in enumerate(names):
        row = {"Attack": nm, "N": int((y == c).sum()), "acc_full": acc0[c]}
        for j in range(K):
            cross_j = sum(alpha[:, k:k + 1] * us[k]
                          for k in range(K) if k != j)
            p = head.predict(np.concatenate([mlp_out, cross_j], 1),
                             batch_size=1024, verbose=0)
            row[f"drop_u{j + 1}"] = acc0[c] - per_class_acc(y, p, n_classes)[c]
        drops = [row[f"drop_u{k + 1}"] for k in range(K)]
        row["critical_degree"] = int(np.argmax(drops)) + 1
        row["alpha_argmax"] = int(np.argmax(alpha[y == c].mean(0))) + 1
        eff_c = alpha[y == c].mean(0) * norms
        row["eff_argmax"] = int(np.argmax(eff_c)) + 1
        row["cross_effect"] = float(np.sum(np.abs(drops)))
        row["agree"] = row["critical_degree"] == row["alpha_argmax"]
        row["agree_eff"] = row["critical_degree"] == row["eff_argmax"]
        rows.append(row)

    ddf = pd.DataFrame(rows)
    print("\n[2] 차수 차단 — u_k 를 끊으면 어느 클래스가 무너지는가")
    print(ddf.round(4).to_string(index=False))
    ddf.to_csv(C.TABLE_DIR / f"occl_degree_{stem}_{a.variant}.csv", index=False)

    print(f"\n[3] 정합성 (전체 {len(ddf)}개 클래스)")
    print(f"    alpha      기준: {ddf['agree'].mean():.1%} "
          f"({int(ddf['agree'].sum())}/{len(ddf)})")
    print(f"    alpha*norm 기준: {ddf['agree_eff'].mean():.1%} "
          f"({int(ddf['agree_eff'].sum())}/{len(ddf)})")

    # 교차 분기 기여가 유의미한 클래스로 한정
    sig = ddf[ddf["cross_effect"] > 0.05]
    if len(sig):
        print(f"\n    교차 기여가 유의미한 클래스만 ({len(sig)}개: "
              f"{', '.join(sig['Attack'])})")
        print(f"    alpha      기준: {sig['agree'].mean():.1%}")
        print(f"    alpha*norm 기준: {sig['agree_eff'].mean():.1%}")
        best = max(sig['agree'].mean(), sig['agree_eff'].mean())
    else:
        best = max(ddf['agree'].mean(), ddf['agree_eff'].mean())

    print()
    if best >= 0.7:
        print("    -> 게이트가 인과적 기여를 반영합니다. 강한 해석 주장 가능.")
    elif best >= 0.5:
        print("    -> 부분적으로 일치. 한계를 명시하며 주장하세요.")
    else:
        print("    -> 게이트와 실제 기여가 어긋납니다.")
        print("       차단 기반 지표(drop_u_k)를 해석 지표로 대체하세요.")


if __name__ == "__main__":
    main()
