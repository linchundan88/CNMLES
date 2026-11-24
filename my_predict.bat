@echo off
rem ============================================================
rem  调用 my_predict.py，分别使用 qwen4-max、gpt-6.1-sol 对
rem  Chinese_Dental_Licensing_Examination.xlsx 逐题预测答案，
rem  结果保存到 results 目录：
rem      results\Chinese_Dental_Licensing_Examination_<模型名>.xlsx
rem ============================================================
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

pushd "%~dp0"

set "PY=python"
%PY% -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到可用的 python 命令，请先确认本机 Python 已加入 PATH。
    popd
    exit /b 1
)

set "RC=0"
for %%M in (qwen4-max gpt-6.1-sol gemini-4-carbon) do (
    echo.
    echo ============================================================
    echo 开始预测，模型：%%M
    echo ============================================================
    %PY% "%~dp0my_predict.py" --model %%M %*
    if errorlevel 1 (
        echo [错误] 模型 %%M 预测失败。
        set "RC=1"
    )
)

echo.
echo ============================================================
echo 全部任务结束，results 目录下的结果文件：
echo ============================================================
dir /b "%~dp0results\*.xlsx"

popd
exit /b %RC%
