call ..\..\..\venv\Scripts\activate.bat
set currDirFullPath=%cd%
for %%a in ("%cd%") do set "currDirName=%%~nxa"
cd ..\..\
:: call python main_fullEval.py "--fe_name=%currDirName%" "--fe_sub_name=alpha1_dd5_dv0__5" "--cl_name=''" "--cl_sub_name=''" "--fe_epoch_num=501" "--cl_epoch_num=0" "--output_path=%currDirFullPath%\alpha1_dd5_dv0__5\initial_check"
:: call python main_fullEval.py "--fe_name=%currDirName%" "--fe_sub_name=alpha1_dd5_dv1" "--cl_name=''" "--cl_sub_name=''" "--fe_epoch_num=501" "--cl_epoch_num=0" "--output_path=%currDirFullPath%\alpha1_dd5_dv1\initial_check"
:: call python main_fullEval.py "--fe_name=%currDirName%" "--fe_sub_name=alpha1_dd10_dv0__5" "--cl_name=''" "--cl_sub_name=''" "--fe_epoch_num=501" "--cl_epoch_num=0" "--output_path=%currDirFullPath%\alpha1_dd10_dv0__5\initial_check"
:: call python main_fullEval.py "--fe_name=%currDirName%" "--fe_sub_name=alpha1_dd10_dv1" "--cl_name=''" "--cl_sub_name=''" "--fe_epoch_num=501" "--cl_epoch_num=0" "--output_path=%currDirFullPath%\alpha1_dd10_dv1\initial_check"
call python main_fullEval.py "--fe_name=%currDirName%" "--fe_sub_name=alpha1_dd20_dv0__5" "--cl_name=''" "--cl_sub_name=''" "--fe_epoch_num=501" "--cl_epoch_num=0" "--output_path=%currDirFullPath%\alpha1_dd20_dv0__5\initial_check"
:: call python main_fullEval.py "--fe_name=%currDirName%" "--fe_sub_name=alpha1_dd20_dv1" "--cl_name=''" "--cl_sub_name=''" "--fe_epoch_num=501" "--cl_epoch_num=0" "--output_path=%currDirFullPath%\alpha1_dd20_dv1\initial_check"
:: call python main_fullEval.py "--fe_name=%currDirName%" "--fe_sub_name=alpha2_dd5_dv0__5" "--cl_name=''" "--cl_sub_name=''" "--fe_epoch_num=501" "--cl_epoch_num=0" "--output_path=%currDirFullPath%\alpha2_dd5_dv0__5\initial_check"
:: call python main_fullEval.py "--fe_name=%currDirName%" "--fe_sub_name=alpha2_dd5_dv1" "--cl_name=''" "--cl_sub_name=''" "--fe_epoch_num=501" "--cl_epoch_num=0" "--output_path=%currDirFullPath%\alpha2_dd5_dv1\initial_check"
call python main_fullEval.py "--fe_name=%currDirName%" "--fe_sub_name=alpha2_dd10_dv0__5" "--cl_name=''" "--cl_sub_name=''" "--fe_epoch_num=501" "--cl_epoch_num=0" "--output_path=%currDirFullPath%\alpha2_dd10_dv0__5\initial_check"
:: call python main_fullEval.py "--fe_name=%currDirName%" "--fe_sub_name=alpha2_dd10_dv1" "--cl_name=''" "--cl_sub_name=''" "--fe_epoch_num=501" "--cl_epoch_num=0" "--output_path=%currDirFullPath%\alpha2_dd10_dv1\initial_check"
:: call python main_fullEval.py "--fe_name=%currDirName%" "--fe_sub_name=alpha2_dd20_dv0__5" "--cl_name=''" "--cl_sub_name=''" "--fe_epoch_num=501" "--cl_epoch_num=0" "--output_path=%currDirFullPath%\alpha2_dd20_dv0__5\initial_check"
:: call python main_fullEval.py "--fe_name=%currDirName%" "--fe_sub_name=alpha2_dd20_dv1" "--cl_name=''" "--cl_sub_name=''" "--fe_epoch_num=501" "--cl_epoch_num=0" "--output_path=%currDirFullPath%\alpha2_dd20_dv1\initial_check"

cd %currDirFullPath%
deactivate
