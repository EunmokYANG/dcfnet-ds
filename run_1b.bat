@echo off
chcp 65001 > nul
set PYTHONUTF8=1
cd /d "%~dp0"

rem ===== 1b : 트리 대조군 (hgb, rf) - 전체 학습 파티션, 다중 시드 =====
rem R3 Comment 2 대응. --max-train 을 주지 않으므로 전체 학습 파티션을 쓴다.
rem 시드 수는 SEEDS 로 조절한다 (R3 요구 최소 3, m4 와 맞추려면 10).
set SEEDS=3

echo ===== 1b 시작 (SEEDS=%SEEDS%) =====
for %%D in (ciciot2023 nfunsw) do (
    for %%S in (minmax standard signedlog signedlog_minmax none) do (
        echo ----- %%D / %%S -----
        python -m src.baselines --dataset %%D --tag full_scaler%%S --models rf --seeds %SEEDS% --seed-start 1
    )
)

echo.
echo ===== 1b 완료 =====
pause
