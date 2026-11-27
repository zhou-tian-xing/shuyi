@echo off
rem Rebuild the CUDA engine from the ASCII path C:\shuyi_src (nvcc rejects the
rem project's non-ASCII directory name).  Needs the MSVC environment, which is why
rem this is a .bat and not a shell one-liner.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cmake --build C:\shuyi_src\build-cuda -j 8
exit /b %errorlevel%
