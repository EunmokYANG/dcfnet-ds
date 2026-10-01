@echo off
chcp 65001 > nul
set PYTHONUTF8=1
cd /d "%~dp0"

rem ===== 1c 사전 측정 : logreg 1회 실행으로 학습 시간과 수렴 여부 확인 =====
rem 가장 오래 걸릴 조건(none: 스케일 없음)과 가장 빠를 조건(signedlog)을 데이터셋마다 1회씩.
for %%D in (ciciot2023 nfunsw) do (
    for %%S in (signedlog none) do (
        echo ----- %%D / %%S -----
        python -m src.baselines --dataset %%D --tag full_scaler%%S --models logreg --seed 1
    )
)
echo.
echo ===== 1c 사전 측정 완료 : 학습 시간(초)과 n_iter 를 확인하세요 =====
pause
