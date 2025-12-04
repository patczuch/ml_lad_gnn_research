@echo off
setlocal

set DATASETS=COLLAB IMDB-BINARY IMDB-MULTI MUTAG ogbg-molhiv PROTEINS REDDIT-BINARY

for %%D in (%DATASETS%) do (
    echo.
    echo ===== Running: eval-models.bat %%D =====
    call eval-models.bat %%D
)

endlocal
echo Done all