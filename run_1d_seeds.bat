@echo off
chcp 65001 > nul
cd /d "%~dp0"
set CUDA_VISIBLE_DEVICES=-1

echo ===== 1d 시드 2 =====
for %%S in (minmax standard signedlog signedlog_minmax none) do (
    python -m src.p1_mechanism_v1 --dataset ciciot2023 --tag full_scaler%%S --seed 2 --skip-perm
    python -m src.p1_mechanism_v1 --dataset nfunsw --tag full_scaler%%S --seed 2 --skip-perm
)

echo ===== 1d 시드 3 =====
for %%S in (minmax standard signedlog signedlog_minmax none) do (
    python -m src.p1_mechanism_v1 --dataset ciciot2023 --tag full_scaler%%S --seed 3 --skip-perm
    python -m src.p1_mechanism_v1 --dataset nfunsw --tag full_scaler%%S --seed 3 --skip-perm
)

echo.
echo ===== 시드 추가 완료 =====
pause