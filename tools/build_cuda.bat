@echo off
rem Rebuild the CUDA engine.
rem
rem nvcc cannot read the project's non-ASCII directory name, so the CUDA build
rem goes through an ASCII path that mirrors the repository:
rem
rem     <CUDA_ROOT>\src     junction -> the repository
rem     <CUDA_ROOT>\build   the CUDA build tree
rem
rem CUDA_ROOT must be an ASCII path *outside* the repository and outside the
rem system drive: the source mirror is not part of the project, and C:\ is not a
rem place to put it.  Override with the SHUYI_CUDA_ROOT environment variable.
rem
rem     tools\build_cuda.bat
rem
rem After this, the CUDA engine is at %CUDA_ROOT%\build\shuyi_engine.exe, and
rem <repo>\build-cuda is a junction to that build tree, so `../build-cuda` still
rem works for engine discovery and the release scripts.
setlocal
set "CUDA_ROOT=%SHUYI_CUDA_ROOT%"
if "%CUDA_ROOT%"=="" set "CUDA_ROOT=F:\projects\shuyi_cuda"
if not exist "%CUDA_ROOT%\src\engine\CMakeLists.txt" (
    echo error: %CUDA_ROOT%\src is not the junction to the repository
    echo create it with: mklink /J "%CUDA_ROOT%\src" ^<path to this repository^>
    exit /b 1
)

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo error: could not set up the MSVC environment
    exit /b 1
)

if not exist "%CUDA_ROOT%\build\CMakeCache.txt" (
    echo configuring %CUDA_ROOT%\build
    cmake -S "%CUDA_ROOT%\src" -B "%CUDA_ROOT%\build" -G Ninja ^
        -DCMAKE_BUILD_TYPE=Release -DSHUYI_CUDA=ON -DSHUYI_CUDA_ARCH=native ^
        -DSHUYI_BUILD_UI=OFF -DSHUYI_BUILD_TOOLS=OFF
    if errorlevel 1 exit /b 1
)

cmake --build "%CUDA_ROOT%\build" -j 8
if errorlevel 1 exit /b 1
echo.
echo CUDA engine: %CUDA_ROOT%\build\shuyi_engine.exe
