@echo off
chcp 65001 > nul
set PYTHONUTF8=1
cd /d "%~dp0"

rem ===== 1c : logreg 대조군 - 전체 학습 파티션 =====
rem lbfgs, 반복 상한 10000 (baselines.py 기본값). 해가 시드에 의존하지 않으므로
rem 조건마다 1회(seed 1) 실행한다 (원고 Appendix A.7).
rem 결정성 확인: CICIoT2023 min-max 를 시드 1-3 으로 돌리면 세 값이 같다 (아래 마지막 줄).

echo ===== 1c 시작 =====
for %%D in (ciciot2023 nfunsw) do (
    for %%S in (minmax standard signedlog signedlog_minmax none) do (
        echo ----- %%D / %%S -----
        python -m src.baselines --dataset %%D --tag full_scaler%%S --models logreg --seed 1
    )
)

rem 결정성 확인 (선택): 앞의 rem 을 지우면 실행된다.
rem python -m src.baselines --dataset ciciot2023 --tag full_scalerminmax --models logreg --seeds 3 --seed-start 1

echo.
echo ===== 1c 완료 =====
pause
