@echo off
rem ============================================================================
rem p2_measure_v1.bat  --  paper 2 measurements, all read-only
rem
rem Place this file in the project root (next to the src folder) and run it
rem from a shell where the dcfnet-ds environment is already active:
rem     p2_measure_v1.bat
rem
rem Everything here only READS. No model is trained, no npz is rebuilt, and
rem no table under outputs\tables is overwritten. The single file written is
rem the log below, under the paper-2 output tree.
rem
rem Output: outputs\paper2_degree\p2_measure_v1.log
rem ============================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "TF_CPP_MIN_LOG_LEVEL=2"
set "TF_ENABLE_ONEDNN_OPTS=0"
set "OUTDIR=outputs\paper2_degree"
set "LOG=%OUTDIR%\p2_measure_v1.log"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"
if exist "%LOG%" del "%LOG%"

echo p2_measure_v1  %DATE% %TIME%>>"%LOG%"

rem ---------------------------------------------------------------- Table III
echo.>>"%LOG%"
echo ==============================================================>>"%LOG%"
echo [1/9] parameters  CICIoT2023  m4  K=4>>"%LOG%"
echo ==============================================================>>"%LOG%"
python -m src.p2_coeff_attribution_v1 --ckpt outputs\models\ciciot2023_full_degree4_m4.keras --summary >>"%LOG%" 2>&1

echo.>>"%LOG%"
echo ==============================================================>>"%LOG%"
echo [2/9] parameters  NF-UNSW-NB15-v3  m4  K=4>>"%LOG%"
echo ==============================================================>>"%LOG%"
python -m src.p2_coeff_attribution_v1 --ckpt outputs\models\nfunsw_full_degree4_m4.keras --summary >>"%LOG%" 2>&1

echo.>>"%LOG%"
echo ==============================================================>>"%LOG%"
echo [3/9] parameters  CICIoT2023  m4g>>"%LOG%"
echo ==============================================================>>"%LOG%"
python -m src.p2_coeff_attribution_v1 --ckpt outputs\models\ciciot2023_full_m4g.keras --summary >>"%LOG%" 2>&1

echo.>>"%LOG%"
echo ==============================================================>>"%LOG%"
echo [4/9] parameters  CICIoT2023  m4m>>"%LOG%"
echo ==============================================================>>"%LOG%"
python -m src.p2_coeff_attribution_v1 --ckpt outputs\models\ciciot2023_full_m4m.keras --summary >>"%LOG%" 2>&1

echo.>>"%LOG%"
echo ==============================================================>>"%LOG%"
echo [5/9] parameters  CICIoT2023  nn_mlp>>"%LOG%"
echo ==============================================================>>"%LOG%"
python -m src.p2_coeff_attribution_v1 --ckpt outputs\models\ciciot2023_full_nn_mlp.keras --summary >>"%LOG%" 2>&1

rem ------------------------------------------------- Table II vs Table III: n
echo.>>"%LOG%"
echo ==============================================================>>"%LOG%"
echo [6/9] seeds  CICIoT2023  degree sweep  (Table II)>>"%LOG%"
echo ==============================================================>>"%LOG%"
python -m src.p2_verify_numbers_v1 seeds --csv outputs\tables\seeds_ciciot2023_full_degree4.csv >>"%LOG%" 2>&1

echo.>>"%LOG%"
echo ==============================================================>>"%LOG%"
echo [7/9] seeds  CICIoT2023  variant comparison  (Table III)>>"%LOG%"
echo ==============================================================>>"%LOG%"
python -m src.p2_verify_numbers_v1 seeds --csv outputs\tables\seeds_ciciot2023_full.csv >>"%LOG%" 2>&1

echo.>>"%LOG%"
echo ==============================================================>>"%LOG%"
echo [8/9] seeds  NF-UNSW-NB15-v3  degree sweep  (Table II)>>"%LOG%"
echo ==============================================================>>"%LOG%"
python -m src.p2_verify_numbers_v1 seeds --csv outputs\tables\seeds_nfunsw_full_degree4.csv >>"%LOG%" 2>&1

echo.>>"%LOG%"
echo ==============================================================>>"%LOG%"
echo [9/9] seeds  NF-UNSW-NB15-v3  variant comparison  (Table III)>>"%LOG%"
echo ==============================================================>>"%LOG%"
python -m src.p2_verify_numbers_v1 seeds --csv outputs\tables\seeds_nfunsw_full.csv >>"%LOG%" 2>&1

echo.
echo ---------------------------------------------------------------
type "%LOG%"
echo ---------------------------------------------------------------
echo saved to %LOG%
endlocal