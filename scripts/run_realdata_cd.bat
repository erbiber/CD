@echo off
setlocal
cd /d "%~dp0"
set PARTS=8
for %%I in (ica noica) do (
  for %%D in (erpcore ds006018) do (
    for /L %%K in (1,1,%PARTS%) do (
      echo === %%D %%I part %%K of %PARTS% ===
      if "%%I"=="ica" (python realdata_cd.py --dataset %%D --part %%K/%PARTS% --jobs 1) else (python realdata_cd.py --dataset %%D --part %%K/%PARTS% --jobs 1 --no-ica)
    )
    echo === %%D %%I: writing output files ===
    if "%%I"=="ica" (python realdata_cd.py --dataset %%D --jobs 1) else (python realdata_cd.py --dataset %%D --jobs 1 --no-ica)
  )
)
endlocal
