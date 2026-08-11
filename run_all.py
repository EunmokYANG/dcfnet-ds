"""전체 파이프라인 일괄 실행.

실행:  python run_all.py --dataset ciciot2023
"""

import argparse
import subprocess
import sys

STEPS = [
    ("전처리", ["-m", "src.preprocess", "--dataset", "{ds}"]),
    ("학습(전체 변형)", ["-m", "src.train", "--dataset", "{ds}", "--variant", "all"]),
    ("평가", ["-m", "src.evaluate", "--dataset", "{ds}"]),
    ("차수 분석", ["-m", "src.degree_analysis", "--dataset", "{ds}", "--variant", "m3"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ciciot2023")
    ds = ap.parse_args().dataset
    for i, (label, args) in enumerate(STEPS, 1):
        cmd = [sys.executable] + [a.format(ds=ds) for a in args]
        print(f"\n{'=' * 60}\n[{i}/{len(STEPS)}] {label}\n{'=' * 60}")
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"[중단] {label} 실패")
    print("\n완료. outputs/ 를 확인하세요.")


if __name__ == "__main__":
    main()
