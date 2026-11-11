@echo off
chcp 65001 > nul
set PYTHONUTF8=1
cd /d "%~dp0"

rem ----- conda environment (TensorFlow 2.16.2 lives only in dcfnet-ds) -----
call "C:\anaconda3\Scripts\activate.bat" dcfnet-ds
python -c "import tensorflow as tf, sys; print('[env]', sys.executable, 'TF', tf.__version__)" 2> nul
if errorlevel 1 (
    echo [stop] TensorFlow not found. Check the conda environment dcfnet-ds.
    pause
    exit /b 1
)

rem ===== 1g : 42-feature seeds 1-3 under the main protocol =====
rem The seed 1-3 checkpoints that run_1e.bat re-evaluated were the 40-epoch budget runs
rem of Table IX ("42 feat., 40 ep."), not the early-stopping runs. This batch
rem   1. moves those checkpoints and histories to outputs\models\budget40 and
rem      outputs\tables\budget40 (Table IX stays reproducible),
rem   2. retrains seeds 1-3 with src\config.py (EPOCHS 500, patience 20), the protocol of seeds 4-10,
rem   3. compares the new seeds 1-3 with backup_catdrop (values printed in the manuscript),
rem   4. recomputes the ten-seed statistics.
rem The 191-feature seeds 1-3 were trained under the main protocol and are not touched.

python -m src.p1_catdrop_protocol_v1 --step isolate
if errorlevel 1 (
    pause
    exit /b 1
)

for %%S in (minmax signedlog signedlog_minmax) do (
    echo ----- nfunsw / catdrop_%%S seeds 1-3 -----
    python -m src.multiseed --dataset nfunsw --tag catdrop_%%S --variants m4 --seeds 3 --seed-start 1
)

python -m src.p1_catdrop_protocol_v1 --step compare
python -m src.p1_encoding10_stats_v1 --seeds 10

echo.
echo ===== 1g done =====
pause
