"""재투고 원고의 [ ] 자리값을 실험 PC 에서 한 번에 모은다.

IEEE Access Access-2026-40899, Appendix A.10 / Code availability / Table III.

출력 항목
  1) Python, NumPy, pandas, scikit-learn, TensorFlow 버전
  2) git 커밋 해시, 태그, 커밋되지 않은 변경 수
  3) NF-UNSW-NB15-v3 시간 분할 경계 (마지막 학습 / 첫 평가 FLOW_START_MILLISECONDS)
  4) Table III 191-feature 행의 양수 시드 수 (seeds_nfunsw_*.csv 중 평균이 맞는 파일 탐색)

실행:
  python -m src.p1_fill_placeholders_v1
"""

import datetime as dt
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from src import config as C

METRIC = "PR-AUC(macro)"
# V5 Table VIII 191-feature 열 (3시드 평균)
TARGET_191 = {"minmax": 0.4731, "signedlog": 0.4889, "signedlog_minmax": 0.5022}


def versions():
    print("\n[1] 버전")
    print(f"  Python       {sys.version.split()[0]}")
    import sklearn
    print(f"  NumPy        {np.__version__}")
    print(f"  pandas       {pd.__version__}")
    print(f"  scikit-learn {sklearn.__version__}")
    try:
        import tensorflow as tf
        print(f"  TensorFlow   {tf.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"  TensorFlow   (불러오기 실패: {e})")


def git_info():
    print("\n[2] git")

    def run(args):
        try:
            return subprocess.check_output(["git"] + args, cwd=C.ROOT,
                                           stderr=subprocess.STDOUT,
                                           text=True).strip()
        except Exception as e:  # noqa: BLE001
            return f"(실패: {e})"
    print(f"  HEAD         {run(['rev-parse', 'HEAD'])}")
    print(f"  describe     {run(['describe', '--tags', '--always'])}")
    print(f"  tags         {run(['tag', '--list']) or '(없음)'}")
    st = run(["status", "--porcelain"])
    n = 0 if not st or st.startswith("(실패") else len(st.splitlines())
    print(f"  미커밋 변경  {n} 개 파일"
          + ("  -> 릴리스 전에 커밋하세요" if n else ""))


def time_cut():
    print("\n[3] NF-UNSW-NB15-v3 시간 분할 경계")
    spec = C.DATASETS["nfunsw"]
    tcol = spec["time_col"]
    cands = [C.RAW_DIR / spec["raw_file"].replace(".csv", "_full.csv"),
             C.RAW_DIR / spec["raw_file"], C.SOURCE_DIR / spec["raw_file"]]
    path = next((p for p in cands if p.exists()), None)
    if path is None:
        print("  원본 CSV 를 찾지 못했습니다:", *map(str, cands), sep="\n    ")
        return
    t = pd.read_csv(path, usecols=[tcol])[tcol].to_numpy()
    order = np.argsort(t, kind="stable")
    cut = int(len(order) * (1 - C.TEST_SIZE))
    last_tr, first_te = t[order[cut - 1]], t[order[cut]]

    def iso(ms):
        return dt.datetime.fromtimestamp(float(ms) / 1000, dt.timezone.utc).isoformat()
    print(f"  파일         {path.name}  ({len(t):,} 행)")
    print(f"  학습 / 평가  {cut:,} / {len(t) - cut:,}")
    print(f"  마지막 학습  {int(last_tr)} ms  ({iso(last_tr)})")
    print(f"  첫 평가      {int(first_te)} ms  ({iso(first_te)})")
    if len(t) != 2_242_931:
        print("  [주의] 행 수가 논문 값 2,242,931 과 다릅니다. 정제 후 파일인지 확인하세요.")


def pos_191():
    print("\n[4] Table III 191-feature 행: 양수 시드 수")
    files = glob.glob(str(C.TABLE_DIR / "seeds_nfunsw_*.csv"))
    found = {}
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        if METRIC not in df.columns or "seed" not in df.columns:
            continue
        if "Variant" in df.columns:
            df = df[df.Variant == "m4"]
        if df.empty:
            continue
        m = float(df[METRIC].mean())
        for sc, target in TARGET_191.items():
            if abs(m - target) < 0.0006 and len(df) >= 3:
                found.setdefault(sc, []).append(
                    (os.path.basename(f), df.set_index("seed")[METRIC]))
    for sc in TARGET_191:
        hits = found.get(sc, [])
        print(f"  {sc:17s} 후보 {len(hits)} 개: " + ", ".join(h[0] for h in hits))
    if not all(found.get(sc) for sc in TARGET_191):
        print("  일치하는 파일을 다 찾지 못했습니다. 아래 목록에서 191-feature 실행 파일명을 알려 주세요.")
        for f in sorted(files):
            print("    ", os.path.basename(f))
        return
    base = found["minmax"][0][1]
    for sc in ("signedlog", "signedlog_minmax"):
        cur = found[sc][0][1]
        common = sorted(set(base.index) & set(cur.index))
        diff = (cur.loc[common] - base.loc[common]) * 100
        print(f"  {sc:17s} - minmax: 시드 {common}, 차이(pp) "
              f"{[round(x, 2) for x in diff]}  -> pos = {int((diff > 0).sum())}/{len(common)}")


def main():
    versions()
    git_info()
    time_cut()
    pos_191()
    print()


if __name__ == "__main__":
    main()
