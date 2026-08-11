@echo off
rem ============================================================================
rem p2_ttest_v1.bat  --  ablation significance tests for Table IV of paper 2
rem Read-only. Writes nothing but the log below.
rem Output: outputs\paper2_degree\p2_ttest_v1.log
rem ============================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "OUTDIR=outputs\paper2_degree"
set "LOG=%OUTDIR%\p2_ttest_v1.log"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
if exist "%LOG%" del "%LOG%"

echo p2_ttest_v1  %DATE% %TIME%>>"%LOG%"

echo.>>"%LOG%"
echo ==============================================================>>"%LOG%"
echo [1/4] CICIoT2023  m4 vs m4g   (gate)>>"%LOG%"
echo ==============================================================>>"%LOG%"
python -m src.p2_verify_numbers_v1 ttest --csv outputs\tables\seeds_ciciot2023_full.csv --a m4 --b m4g --family 4 >>"%LOG%" 2>&1

echo.>>"%LOG%"
echo ==============================================================>>"%LOG%"
echo [2/4] CICIoT2023  m4 vs m4m   (MLP branch)>>"%LOG%"
echo ==============================================================>>"%LOG%"
python -m src.p2_verify_numbers_v1 ttest --csv outputs\tables\seeds_ciciot2023_full.csv --a m4 --b m4m --family 4 >>"%LOG%" 2>&1

echo.>>"%LOG%"
echo ==============================================================>>"%LOG%"
echo [3/4] NF-UNSW-NB15-v3  m4 vs m4g   (gate)>>"%LOG%"
echo ==============================================================>>"%LOG%"
python -m src.p2_verify_numbers_v1 ttest --csv outputs\tables\seeds_nfunsw_full.csv --a m4 --b m4g --family 4 >>"%LOG%" 2>&1

echo.>>"%LOG%"
echo ==============================================================>>"%LOG%"
echo [4/4] NF-UNSW-NB15-v3  m4 vs m4m   (MLP branch)>>"%LOG%"
echo ==============================================================>>"%LOG%"
python -m src.p2_verify_numbers_v1 ttest --csv outputs\tables\seeds_nfunsw_full.csv --a m4 --b m4m --family 4 >>"%LOG%" 2>&1

echo.
echo ---------------------------------------------------------------
type "%LOG%"
echo ---------------------------------------------------------------
echo saved to %LOG%
endlocal