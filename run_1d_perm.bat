@echo off
chcp 65001 > nul
cd /d "%~dp0"
set CUDA_VISIBLE_DEVICES=-1

python -m src.p1_mechanism_v1 --dataset ciciot2023 --tag full_scalerminmax
python -m src.p1_mechanism_v1 --dataset ciciot2023 --tag full_scalersignedlog
python -m src.p1_mechanism_v1 --dataset nfunsw --tag full_scalerminmax
python -m src.p1_mechanism_v1 --dataset nfunsw --tag full_scalersignedlog
pause