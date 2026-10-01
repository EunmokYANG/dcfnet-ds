@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ===== 1a m4 시드 확대 3 -^> 10 (4조건 x 2데이터셋) =====
echo     --reuse : 시드 1~3 은 체크포인트 재평가, 4~10 만 학습
echo.

for %%S in (minmax signedlog signedlog_minmax none) do (
    echo ----- ciciot2023 / %%S -----
    python -m src.multiseed --dataset ciciot2023 --tag full_scaler%%S --variants m4 --seeds 10 --seed-start 1 --reuse
)

for %%S in (minmax signedlog signedlog_minmax none) do (
    echo ----- nfunsw / %%S -----
    python -m src.multiseed --dataset nfunsw --tag full_scaler%%S --variants m4 --seeds 10 --seed-start 1 --reuse
)

echo.
echo ===== 1a 완료 =====
pause