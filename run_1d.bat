@echo off
chcp 65001 > nul
set PYTHONUTF8=1
cd /d "%~dp0"

rem ===== 1d : 메커니즘 측정 (R4 Comment 2) =====
rem 저장된 m4 체크포인트만 사용한다 (재학습 없음).
rem 주 대비 두 조건(min-max, signed-log) x 두 데이터셋 x 시드 1~3, permutation 3회 반복.
set REPEATS=3

echo ===== 1d 시작 =====
for %%D in (ciciot2023 nfunsw) do (
    for %%S in (minmax signedlog) do (
        for %%N in (1 2 3) do (
            echo ----- %%D / %%S / seed %%N -----
            python -m src.p1_mechanism_v1 --dataset %%D --tag full_scaler%%S --seed %%N --repeats %REPEATS%
        )
    )
)

echo.
echo ===== 교차 스케일러 비교 =====
python -m src.p1_mechanism_compare_v1 --dataset ciciot2023
python -m src.p1_mechanism_compare_v1 --dataset nfunsw
python -m src.p1_mechanism_compare_v1 --dataset nfunsw --exclude-categorical

echo.
echo ===== 1d 완료 =====
pause