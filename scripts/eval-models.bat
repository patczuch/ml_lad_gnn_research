@echo off
setlocal enabledelayedexpansion

rem --- Check dataset argument ---
if "%~1"=="" (
    echo Usage: %0 <dataset> [optional additional arguments]
    exit /b 1
)

set DATASET=%1
set EXTRA=
if not "%2"=="" (
    set EXTRA=
    for %%A in (%*) do (
        if NOT "%%A"=="%1" set EXTRA=!EXTRA! %%A
    )
)

set BACKBONES=GAT GCN GIN GraphSAGE
set MODES=T S P

for %%B in (%BACKBONES%) do (
    for %%M in (%MODES%) do (
        echo =======================================
        echo Starting: backbone=%%B ^| train_mode=%%M
        echo ---------------------------------------

        python ..\main.py --dataset %DATASET% --backbone %%B --train_mode %%M %EXTRA%
        )
    )

endlocal

echo Done