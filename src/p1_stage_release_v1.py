"""GitHub v2.0.1 업로드용 파일을 모은다 (원본은 건드리지 않고 복사만 한다).

모으는 것
  outputs/tables/*.csv, *.xlsx            --since 이후 수정된 결과 파일 (10 시드, 전체 파티션 대조군 등)
  outputs/figures/paper1_scaler/*_resub.*  재투고 Fig. 2, Fig. 5
  requirements-lock.txt                   pip freeze
  environment-lock.yml                    conda env export --no-builds (conda 가 있을 때)
  MANIFEST_sha256.txt                     위 파일 전체의 SHA-256 (GitHub / Zenodo / 제출 ZIP 동일성 확인용)

결과 : _release_v2.0.1/batch1, batch2, ...  (GitHub 웹 업로드는 한 번에 100개까지라 90개씩 나눈다)
       각 batchN 폴더 안의 내용물(outputs 폴더 등)을 GitHub 업로드 화면에 끌어다 놓으면 경로가 유지된다.
실행 : python -m src.p1_stage_release_v1
       python -m src.p1_stage_release_v1 --since 2026-08-11
"""

import argparse
import datetime as dt
import hashlib
import shutil
import subprocess
import sys

from src import config as C

BATCH = 90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-08-11", help="이 날짜(포함) 이후 수정된 결과만 모은다")
    ap.add_argument("--out", default="_release_v2.0.1")
    a = ap.parse_args()
    t0 = dt.datetime.fromisoformat(a.since).timestamp()
    root = C.ROOT
    out = root / a.out
    if out.exists():
        sys.exit(f"[중단] {out} 가 이미 있습니다. 지우거나 --out 으로 다른 이름을 주세요 (덮어쓰지 않음).")

    files = []
    for pat in ("*.csv", "*.xlsx"):
        files += [f for f in C.TABLE_DIR.glob(pat) if f.stat().st_mtime >= t0]
    figdir = C.FIG_DIR / "paper1_scaler"
    if figdir.is_dir():
        files += sorted(figdir.glob("*_resub.*"))
    files = sorted(set(files))

    lock = []
    r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    if r.returncode == 0:
        (root / "requirements-lock.txt").write_text(r.stdout, encoding="utf-8"); lock.append(root / "requirements-lock.txt")
    try:
        r = subprocess.run("conda env export --no-builds", shell=True, capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            txt = "\n".join(l for l in r.stdout.splitlines() if not l.startswith("prefix:")) + "\n"
            (root / "environment-lock.yml").write_text(txt, encoding="utf-8"); lock.append(root / "environment-lock.yml")
    except Exception as e:
        print(f"[참고] conda env export 실패: {e}")
    files += lock

    man = []
    for i, f in enumerate(files):
        rel = f.relative_to(root)
        dst = out / f"batch{i // BATCH + 1}" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        man.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {rel.as_posix()}")
    last = out / f"batch{(len(files) - 1) // BATCH + 1}"
    (last / "MANIFEST_sha256.txt").write_text("\n".join(man) + "\n", encoding="utf-8")
    shutil.copy2(last / "MANIFEST_sha256.txt", out / "MANIFEST_sha256.txt")

    print(f"[완료] {len(files)}개 파일 -> {out}")
    for b in sorted(out.glob("batch*")):
        n = sum(1 for p in b.rglob('*') if p.is_file())
        print(f"   {b.name}: {n}개")
    print("   잠금 파일:", ", ".join(p.name for p in lock) or "없음")
    print("   목록: MANIFEST_sha256.txt")


if __name__ == "__main__":
    main()
