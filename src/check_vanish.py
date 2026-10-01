"""특징별 크기(RMS) 격차와 소멸 특징 수 확인.

사용법: python check_vanish.py <stem>
예:     python check_vanish.py nfunsw_full_scalernone
"""
import sys

import numpy as np

from src import config as C

stem = sys.argv[1]
d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
x = d["x_train"][:300000].astype("float64")
names = list(d["feature_names"])
r = np.sqrt((x ** 2).mean(axis=0))
p = r[r > 0]
n_vanish = int((r / r.max() < 1e-4).sum())

print(f"\n{stem}")
print(f"  특징 수            : {x.shape[1]}")
print(f"  RMS 최대/최소      : {p.max() / p.min():.3g}")
print(f"  소멸 특징(최대의 1e-4 미만): {n_vanish}개")
top = np.argsort(-r)[:5]
print("  RMS 상위 5개       : " + ", ".join(f"{names[j]}({r[j]:.3g})" for j in top))