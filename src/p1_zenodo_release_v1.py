"""Zenodo 새 버전(v2.0.0) 만들기: 코드 스냅숏 zip 생성 -> 업로드 -> 메타데이터 -> (선택) 게시.

IEEE Access Access-2026-40899 재투고, 기존 레코드 10.5281/zenodo.21883691 의 새 버전.

준비
  1) Zenodo 로그인 -> 오른쪽 위 계정 -> Applications -> Personal access tokens -> New token
     권한(scopes): deposit:write, deposit:actions 체크 -> 토큰 복사
  2) 명령창에서:  set ZENODO_TOKEN=복사한토큰

실행 (3단계, 앞 단계 결과를 확인한 뒤 다음 단계로)
  python -m src.p1_zenodo_release_v1 --zip-only     # zip 만 만들고 내용 목록 출력
  python -m src.p1_zenodo_release_v1 --draft        # 새 버전 초안 생성 + 업로드 + 메타데이터 (게시 안 함)
  python -m src.p1_zenodo_release_v1 --publish      # 초안을 게시하고 새 DOI 출력 (되돌릴 수 없음)

--draft 후 Zenodo 웹의 Upload 목록에서 초안을 직접 확인할 수 있다.
게시(publish)하면 DOI 가 확정되고 파일은 더 이상 바꿀 수 없다.
"""

import argparse
import datetime as dt
import json
import os
import sys
import zipfile
from pathlib import Path

from src import config as C

BASE = "https://zenodo.org/api"
RECORD_V1 = "21883691"          # 10.5281/zenodo.21883691
VERSION = "v2.0.0"
ZIP_NAME = f"dcfnet-ds_{VERSION}.zip"
STATE = C.OUT_DIR / "zenodo_draft.json"

INCLUDE_TOP = ["main.py", "run_all.py", "run_experiment.py", "requirements.txt",
               "environment.yml", "README.md", "LICENSE"]
INCLUDE_GLOB_TOP = ["*.bat"]
INCLUDE_DIRS = ["src", "configs", "config"]
EXCLUDE_PARTS = {"__pycache__", ".git", ".idea", "data", "outputs", ".ipynb_checkpoints"}

DESCRIPTION = (
    "<p>Code, configuration files, and analysis scripts accompanying the IEEE Access resubmission "
    "Access-2026-40899, &ldquo;Feature Scaling in Arithmetic-Combination Intrusion Detectors: "
    "Empirical Limits of Affine Normalization&rdquo;.</p>"
    "<p>This version adds the ten-seed primary runs, full-partition control models "
    "(HistGradientBoosting, random forest, logistic regression), paired statistics with Holm "
    "correction within dataset, a feature-level attribution analysis, and the script that builds "
    "the supplementary per-seed workbook. Tables and figures not covered by the new scripts are "
    "produced by the unchanged v1.0.0 scripts included here.</p>"
    "<p>Environment: Python 3.11, TensorFlow 2.16.2 (CPU), NumPy 1.26, scikit-learn 1.9. "
    "Datasets are not redistributed; see src/prepare_data.py and the manuscript Appendix A.</p>"
)


def build_zip():
    root = Path(C.ROOT)
    out = C.OUT_DIR / ZIP_NAME
    files = []
    for name in INCLUDE_TOP:
        if (root / name).is_file():
            files.append(root / name)
    for pat in INCLUDE_GLOB_TOP:
        files += sorted(root.glob(pat))
    for d in INCLUDE_DIRS:
        p = root / d
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (set(f.relative_to(root).parts) & EXCLUDE_PARTS) \
                        and f.suffix not in {".pyc", ".npz", ".csv", ".h5", ".keras"}:
                    files.append(f)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path(f"dcfnet-ds-{VERSION}") / f.relative_to(root))
    mb = out.stat().st_size / 1024 / 1024
    print(f"[zip] {out}  ({len(files)} files, {mb:.2f} MB)")
    for f in files:
        print("   ", f.relative_to(root))
    return out


def session():
    try:
        import requests
    except ImportError:
        sys.exit("[중단] requests 가 없습니다: pip install requests")
    tok = os.environ.get("ZENODO_TOKEN")
    if not tok:
        sys.exit("[중단] 토큰이 없습니다. 먼저  set ZENODO_TOKEN=토큰  을 실행하세요.")
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {tok}"
    return s


def check(r, what):
    if r.status_code >= 400:
        sys.exit(f"[실패] {what}: HTTP {r.status_code}\n{r.text[:1500]}")
    return r.json() if r.content else {}


def latest_deposition_id(s):
    rec = check(s.get(f"{BASE}/records/{RECORD_V1}"), "기존 레코드 조회")
    latest = rec.get("links", {}).get("latest", "")
    rid = latest.rstrip("/").split("/")[-1] if latest else RECORD_V1
    print(f"[info] 기존 레코드 {RECORD_V1}, 최신 버전 레코드 {rid}")
    return rid


def draft(zip_path):
    s = session()
    rid = latest_deposition_id(s)
    nv = check(s.post(f"{BASE}/deposit/depositions/{rid}/actions/newversion"), "새 버전 생성")
    draft_url = nv["links"]["latest_draft"]
    d = check(s.get(draft_url), "초안 조회")
    did = d["id"]
    print(f"[draft] 초안 id {did}")

    for f in d.get("files", []):                      # v1 에서 상속된 파일 제거
        check(s.delete(f"{BASE}/deposit/depositions/{did}/files/{f['id']}"),
              f"상속 파일 삭제 {f.get('filename')}")
        print(f"   상속 파일 삭제: {f.get('filename')}")

    bucket = d["links"]["bucket"]
    with open(zip_path, "rb") as fh:
        check(s.put(f"{bucket}/{zip_path.name}", data=fh), "zip 업로드")
    print(f"[upload] {zip_path.name}")

    md = d["metadata"]
    md["version"] = VERSION
    md["publication_date"] = dt.date.today().isoformat()
    md["description"] = DESCRIPTION
    md["title"] = ("dcfnet-ds: code for \"Feature Scaling in Arithmetic-Combination Intrusion "
                   "Detectors: Empirical Limits of Affine Normalization\"")
    rel = [x for x in md.get("related_identifiers", [])
           if "github.com/EunmokYANG/dcfnet-ds" not in x.get("identifier", "")]
    rel.append({"identifier": "https://github.com/EunmokYANG/dcfnet-ds",
                "relation": "isSupplementTo", "scheme": "url"})
    md["related_identifiers"] = rel
    md.pop("doi", None)
    md.pop("prereserve_doi", None)
    r = check(s.put(f"{BASE}/deposit/depositions/{did}",
                    data=json.dumps({"metadata": md}),
                    headers={"Content-Type": "application/json"}), "메타데이터 저장")
    pre = r.get("metadata", {}).get("prereserve_doi", {}).get("doi")
    STATE.write_text(json.dumps({"draft_id": did, "prereserved_doi": pre}, indent=2),
                     encoding="utf-8")
    print(f"[ok] 초안 저장 완료 (게시 전). 예약 DOI: {pre}")
    print(f"     확인: https://zenodo.org/uploads/{did}")
    print("     문제가 없으면  python -m src.p1_zenodo_release_v1 --publish")


def publish():
    if not STATE.exists():
        sys.exit("[중단] 초안 정보가 없습니다. 먼저 --draft 를 실행하세요.")
    st = json.loads(STATE.read_text(encoding="utf-8"))
    s = session()
    r = check(s.post(f"{BASE}/deposit/depositions/{st['draft_id']}/actions/publish"), "게시")
    print(f"[published] 새 버전 DOI : {r.get('doi')}")
    print(f"            레코드      : {r.get('links', {}).get('record_html') or r.get('links', {}).get('html')}")
    print(f"            concept DOI : {r.get('conceptdoi')}  (원고에는 위의 버전 DOI 를 쓴다)")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip-only", action="store_true")
    g.add_argument("--draft", action="store_true")
    g.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    if a.zip_only:
        build_zip()
    elif a.draft:
        draft(build_zip())
    else:
        publish()


if __name__ == "__main__":
    main()