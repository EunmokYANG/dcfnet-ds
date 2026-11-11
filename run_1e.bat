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

rem ===== 1e : NF-UNSW-NB15-v3 feature-encoding check, seeds 3 -> 10 =====
rem Access-2026-52475 R1-3, R2-3, R2-4.
rem   catdrop = 42 features (nominal codes and bitmasks dropped)
rem   catenc  = 191 features (one-hot + per-bit encoding, passed through unscaled)
rem --reuse : seeds 1-3 are re-evaluated from the saved checkpoints, seeds 4-10 are trained.
rem Split, validation hold-out (stratified 20 %%, seed 1) and training settings are the
rem same as the 47-feature runs of run_1a.bat (src/config.py, src/train.py unchanged).

rem ----- 0. keep a copy of the three-seed files before they are merged (first run only) -----
set ARC=outputs\tables\_archive_1e_before
if not exist "%ARC%" mkdir "%ARC%"
for %%E in (catdrop catenc) do (
    for %%S in (minmax signedlog signedlog_minmax) do (
        if not exist "%ARC%\seeds_nfunsw_%%E_%%S.csv" if exist "outputs\tables\seeds_nfunsw_%%E_%%S.csv" copy /Y "outputs\tables\seeds_nfunsw_%%E_%%S.csv" "%ARC%\" > nul
        if not exist "%ARC%\summary_nfunsw_%%E_%%S.csv" if exist "outputs\tables\summary_nfunsw_%%E_%%S.csv" copy /Y "outputs\tables\summary_nfunsw_%%E_%%S.csv" "%ARC%\" > nul
    )
)
echo [archive] %ARC%

rem ----- 1. 42 features (short runs first) -----
for %%S in (minmax signedlog signedlog_minmax) do (
    echo ----- nfunsw / catdrop_%%S -----
    python -m src.multiseed --dataset nfunsw --tag catdrop_%%S --variants m4 --seeds 10 --seed-start 1 --reuse
)

rem ----- 2. 191 features (long runs) -----
for %%S in (minmax signedlog signedlog_minmax) do (
    echo ----- nfunsw / catenc_%%S -----
    python -m src.multiseed --dataset nfunsw --tag catenc_%%S --variants m4 --seeds 10 --seed-start 1 --reuse
)

rem ----- 3. (optional) no scaling, to match the four principal conditions -----
rem No checkpoints or npz exist for these yet: preprocessing + 10 seeds each.
rem Remove "rem" from the six lines below to run them.
rem python -m src.preprocess --dataset nfunsw --full --split temporal --scaler none --categorical drop --tag catdrop_none
rem python -m src.preprocess --dataset nfunsw --full --split temporal --scaler none --categorical encode --tag catenc_none
rem for %%E in (catdrop catenc) do (
rem     python -m src.multiseed --dataset nfunsw --tag %%E_none --variants m4 --seeds 10 --seed-start 1
rem )

rem ----- 4. statistics -----
python -m src.p1_encoding10_stats_v1 --seeds 10

echo.
echo ===== 1e done =====
pause
