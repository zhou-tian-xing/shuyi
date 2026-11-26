@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
if errorlevel 1 exit /b 1

rem Use the ASCII source junction and build tree required by nvcc on Windows.
set "SHUYI_CUDA_BUILD=C:\shuyi_cuda_build"
cmake -S C:\shuyi_src -B "%SHUYI_CUDA_BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSHUYI_CUDA=ON -DSHUYI_BUILD_UI=ON
if errorlevel 1 exit /b 1
cmake --build "%SHUYI_CUDA_BUILD%" -j 8
if errorlevel 1 exit /b 1

for %%F in (shuyi.exe shuyi_engine.exe shuyi_ui_tests.exe shuyi_tests.exe) do (
    cmake -E copy_if_different "%SHUYI_CUDA_BUILD%\%%F" "%~dp0%%F"
    if errorlevel 1 exit /b 1
)
exit /b 0
