"""클래스별 재현율·정밀도·PR-AUC 확인 (재학습 없음).

사용법: python check_perclass.py <stem> [seed] [variant]
예:     python check_perclass.py nfunsw_full_scalersignedlog 10
"""
import sys

import numpy as np
from tensorflow.keras import models

from src import config as C
from src.evaluate import per_class_report
from src.model import CUSTOM_OBJECTS

stem = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 10
variant = sys.argv[3] if len(sys.argv) > 3 else "m4"
suffix = "" if seed == C.SEED else f"_s{seed}"
path = C.MODEL_DIR / f"{stem}_{variant}{suffix}{C.MODEL_EXT}"

d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
m = models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
p = m.predict(d["x_test"], batch_size=4096, verbose=0)
rep = per_class_report(d["y_test"], p, C.DATASETS[stem.split("_")[0]]["classes"])

print(f"\nmodel: {path.name}")
print(rep.round(4).to_string())
acc = float((p.argmax(1) == d["y_test"]).mean())
b = rep.loc[rep["Attack"] == "Benign", "Recall"]
print(f"\nAccuracy = {acc:.4f}   Benign recall = {float(b.iloc[0]):.4f}")