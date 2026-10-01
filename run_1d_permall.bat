@echo off
chcp 65001 > nul
cd /d "%~dp0"
set CUDA_VISIBLE_DEVICES=-1

for %%D in (1 2 3) do (
  for %%S in (minmax standard signedlog signedlog_minmax none) do (
    python -m src.p1_mechanism_v1 --dataset ciciot2023 --tag full_scaler%%S --seed %%D
    python -m src.p1_mechanism_v1 --dataset nfunsw --tag full_scaler%%S --seed %%D
  )
)
pause