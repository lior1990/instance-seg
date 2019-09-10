call ..\..\..\venv\Scripts\activate.bat
set outputPath=%cd%
for %%a in ("%cd%") do set "expName=%%~nxa"
cd ..\..\
call python main_fullEval.py "--fe_name=%expName%" "--cl_name=%expName%" "--fe_epoch_num=501" "--cl_epoch_num=5001" "--output_path=%outputPath%"

cd %currDirFullPath%
deactivate
