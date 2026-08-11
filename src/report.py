"""outputs/tables 의 모든 CSV 를 엑셀 한 파일로 통합한다.

논문 표를 만들 때 흩어진 CSV 를 다시 합칠 필요가 없도록, 시트별로
정리하고 서식을 적용한다. 결과 보존 목적도 겸한다.

시트 구성
  00_README        생성 시각, 데이터 프로토콜, 시트 안내
  01_Performance   모델별 성능 (제안 + baseline)
  02_SeedStats     다중 시드 평균 +- 표준편차
  03_SeedRaw       시드별 원시값
  04_Significance  쌍별 Welch t-검정
  05_PerClass      클래스별 재현율·정밀도·PR-AUC
  06_DegreeContrib 차수별 로짓 기여 비중
  07_Interactions  피처 조합 계수 (주 기여)
  08_Perturbation  섭동 검증
  09_Purity        동차성 검정
  10_Config        실험 설정

실행:  python -m src.report
       python -m src.report --out my_results.xlsx
"""

import argparse
import datetime as dt
import json

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from src import config as C

FONT = "Arial"
HDR_FILL = PatternFill("solid", fgColor="1A3A5C")
HDR_FONT = Font(name=FONT, bold=True, color="FFFFFF", size=10)
BODY = Font(name=FONT, size=10)

# (시트명, 파일 접두사, 설명)
SHEETS = [
    ("01_Performance", ["metrics_all_", "metrics_baseline_", "metrics_"],
     "모델별 성능 (단일 시드)"),
    ("02_SeedStats", ["summary_"], "다중 시드 평균 +- 표준편차"),
    ("03_SeedRaw", ["seeds_"], "시드별 원시 지표"),
    ("04_Significance", ["compare_"], "쌍별 Welch t-검정"),
    ("05_PerClass", ["perclass_"], "클래스별 재현율/정밀도/PR-AUC"),
    ("06_DegreeContrib", ["degree_contrib_", "degree_alpha_"],
     "차수별 로짓 기여 비중"),
    ("07_Interactions", ["interactions_"], "피처 조합 계수 (주 기여)"),
    ("08_Perturbation", ["perturb_"], "섭동 검증 (인과성)"),
    ("09_Purity", ["purity_"], "동차성 검정 (차수 분리 증명)"),
    ("11_SweepDegree", ["sweep_degree"], "차수 상한 K 스윕"),
    ("12_SweepScaler", ["sweep_scaler"], "스케일러 스윕"),
    ("13_SweepOther", ["sweep_split", "sweep_amplify"], "분할/증폭 스윕"),
]


def collect(prefixes):
    """접두사에 맞는 CSV 를 모아 dataset/variant 열을 붙여 합친다."""
    frames = []
    for pre in prefixes:
        for p in sorted(C.TABLE_DIR.glob(f"{pre}*.csv")):
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if df.empty:
                continue
            src = p.stem[len(pre):]
            df.insert(0, "source", src)
            frames.append(df)
        if frames:      # 우선순위가 높은 접두사에서 찾으면 중단
            break
    return pd.concat(frames, ignore_index=True) if frames else None


def style(ws, df):
    for j, col in enumerate(df.columns, 1):
        cell = ws.cell(row=1, column=j)
        cell.fill, cell.font = HDR_FILL, HDR_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        width = max(len(str(col)) + 2,
                    int(df[col].astype(str).str.len().quantile(0.9)) + 2)
        ws.column_dimensions[get_column_letter(j)].width = min(width, 52)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = BODY
            if isinstance(cell.value, float):
                cell.number_format = "0.0000"
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def readme_frame(found):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = [
        ("생성 시각", now, ""),
        ("", "", ""),
        ("--- 데이터 프로토콜 ---", "", ""),
        ("스케일러", C.SCALER, "학습 파라미터 없음 -> 배포 시 재적합 불필요"),
        ("분할", C.SPLIT, "NF 는 시간 분할, CICIoT2023 은 무작위"),
        ("분포", "원본 그대로", "축소 샘플링 없음"),
        ("차수 상한 K", str(C.MAX_DEGREE), ""),
        ("에폭 상한", str(C.EPOCHS), f"조기종료 인내 {C.EARLY_STOP_PATIENCE}"),
        ("배치", str(C.BATCH_SIZE), ""),
        ("", "", ""),
        ("--- 시트 안내 ---", "", ""),
    ]
    for name, _, desc in SHEETS:
        rows.append((name, "수록" if name in found else "없음", desc))
    return pd.DataFrame(rows, columns=["항목", "값", "설명"])


def config_frame():
    rows = [("SEED", C.SEED), ("TEST_SIZE", C.TEST_SIZE),
            ("VAL_SIZE", C.VAL_SIZE), ("SCALER", C.SCALER),
            ("SPLIT", C.SPLIT), ("MAX_DEGREE", C.MAX_DEGREE),
            ("EPOCHS", C.EPOCHS), ("BATCH_SIZE", C.BATCH_SIZE),
            ("LEARNING_RATE", C.LEARNING_RATE),
            ("EARLY_STOP_PATIENCE", C.EARLY_STOP_PATIENCE),
            ("FOCAL_GAMMA", C.FOCAL_GAMMA), ("FOCAL_ALPHA", C.FOCAL_ALPHA)]
    for k, v in C.VARIANTS.items():
        rows.append((f"VARIANT.{k}", v["desc"]))
    for name in C.DATASETS:
        p = C.PROCESSED_DIR / f"{name}_full_meta.json"
        if p.exists():
            m = json.loads(p.read_text("utf-8"))
            rows += [(f"{name}.n_features", m["n_features"]),
                     (f"{name}.n_classes", m["n_classes"]),
                     (f"{name}.train", str(m["train_shape"])),
                     (f"{name}.test", str(m["test_shape"])),
                     (f"{name}.scaler", m.get("scaler")),
                     (f"{name}.split", m.get("split"))]
    return pd.DataFrame(rows, columns=["항목", "값"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dcfnet_ds_results.xlsx")
    a = ap.parse_args()
    out = C.OUT_DIR / a.out

    data, found = {}, set()
    for name, prefixes, _ in SHEETS:
        df = collect(prefixes)
        if df is not None:
            data[name], _ = df, found.add(name)

    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        readme_frame(found).to_excel(xw, sheet_name="00_README", index=False)
        for name, _, _ in SHEETS:
            if name in data:
                data[name].to_excel(xw, sheet_name=name, index=False)
        config_frame().to_excel(xw, sheet_name="10_Config", index=False)

        for ws in xw.book.worksheets:
            df = (readme_frame(found) if ws.title == "00_README"
                  else config_frame() if ws.title == "10_Config"
                  else data.get(ws.title))
            if df is not None:
                style(ws, df)

    print(f"\n{'=' * 62}\n  엑셀 리포트 생성 완료\n{'=' * 62}")
    print(f"  경로: {out}")
    print(f"  시트: {len(found) + 2}개")
    for name, _, desc in SHEETS:
        mark = "O" if name in found else "X"
        n = len(data[name]) if name in data else 0
        print(f"    [{mark}] {name:<18}{n:>7,}행   {desc}")
    print()


if __name__ == "__main__":
    main()
