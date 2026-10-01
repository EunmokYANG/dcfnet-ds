@echo off
chcp 65001 > nul
set CUDA_VISIBLE_DEVICES=-1

echo ===== 1d 빠른 스캔 (permutation 생략) =====
for %%S in (minmax standard signedlog signedlog_minmax none) do (
    echo.
    echo ----- ciciot2023 / %%S -----
    python -m src.p1_mechanism_v1 --dataset ciciot2023 --tag full_scaler%%S --skip-perm
    echo.
    echo ----- nfunsw / %%S -----
    python -m src.p1_mechanism_v1 --dataset nfunsw --tag full_scaler%%S --skip-perm
)
echo.
echo ===== 스캔 완료 =====
pause