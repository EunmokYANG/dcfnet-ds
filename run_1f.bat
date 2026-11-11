@echo off
chcp 65001 > nul
set PYTHONUTF8=1
cd /d "%~dp0"

rem ----- conda environment (same package versions as requirements-lock.txt) -----
call "C:\anaconda3\Scripts\activate.bat" dcfnet-ds
python -c "import scipy, sys; print('[env]', sys.executable, 'scipy', scipy.__version__)"

rem ===== 1f : resolution vs distribution shape (Access-2026-52475 R1-4, R2-2) =====
rem No training. Reads the 1d outputs (mech_compare_features_*.csv) and x_train of the
rem min-max, signed-log and no-scaling npz files. Can run while run_1e.bat is training.

rem shape statistics are recomputed in this environment on every run
for %%D in (ciciot2023 nfunsw) do if exist "outputs\tables\mech_shape_%%D.csv" del "outputs\tables\mech_shape_%%D.csv"

python -m src.p1_mechanism_compare_v2 --dataset ciciot2023
python -m src.p1_mechanism_compare_v2 --dataset nfunsw
python -m src.p1_mechanism_compare_v2 --dataset nfunsw --exclude-categorical

echo.
echo ===== 1f done =====
pause
