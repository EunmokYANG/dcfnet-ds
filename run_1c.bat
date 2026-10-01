@echo off
chcp 65001 > nul
set PYTHONUTF8=1
cd /d "%~dp0"

rem ===== 1c : logreg 대조군 - 전체 학습 파티션, 다중 시드 =====
rem solver 는 lbfgs 그대로, 반복 상한만 10000 (baselines.py 기본값).
rem run_1c_probe.bat 결과를 보고 시간이 감당되면 실행한다.
set SEEDS=3

echo ===== 1c 시작 (SEEDS=%SEEDS%) =====
for %%D in (ciciot2023 nfunsw) do (
    for %%S in (minmax standard signedlog signedlog_minmax none) do (
        echo ----- %%D / %%S -----
        python -m src.baselines --dataset %%D --tag full_scaler%%S --models logreg --seeds %SEEDS% --seed-start 1
    )
)

echo.
echo ===== 1c 완료 =====
pause