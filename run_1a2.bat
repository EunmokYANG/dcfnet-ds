@echo off
chcp 65001 > nul
cd /d "%~dp0"

REM 상위 창에서 -1이 설정되어 있어도 상속되지 않도록 제거한다.
set "CUDA_VISIBLE_DEVICES="
echo CUDA_VISIBLE_DEVICES=[%CUDA_VISIBLE_DEVICES%]  (비어 있어야 정상)

REM ===== 0순위: CPU로 학습된 s10을 GPU로 교체 (약 3분) =====
python -m src.multiseed --dataset ciciot2023 --tag full_scalersignedlog_minmax --variants m4 --seeds 10 --seed-start 1 --reuse

REM ===== 1순위: 명제가 이름을 부른 아핀 조건 =====
for %%S in (standard robust_safe) do (
    python -m src.multiseed --dataset ciciot2023 --tag full_scaler%%S --variants m4 --seeds 10 --seed-start 1 --reuse
    python -m src.multiseed --dataset nfunsw --tag full_scaler%%S --variants m4 --seeds 10 --seed-start 1 --reuse
)

REM ===== 2순위: 비선형 비교군 =====
for %%S in (asinh yeojohnson quantile winsor_minmax) do (
    python -m src.multiseed --dataset ciciot2023 --tag full_scaler%%S --variants m4 --seeds 10 --seed-start 1 --reuse
    python -m src.multiseed --dataset nfunsw --tag full_scaler%%S --variants m4 --seeds 10 --seed-start 1 --reuse
)
pause