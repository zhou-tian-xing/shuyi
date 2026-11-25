# 数弈 — developer umbrella build.
#
# The repository holds three independent products and one developer-only tree:
#
#   core/      the rules (linked by both products, and by nothing else)
#   engine/    shuyi_engine.exe   — headless NN-MCTS service
#   ui/        shuyi.exe          — native Win32 front end
#   protocol/  the contract between them (documentation + conformance tests)
#   tools/     developer tools (bench, probes, self-play, tournaments) — never shipped
#   training/  the Python training pipeline — never shipped
#
# This file only exists for convenience.  Each product also builds alone:
#
#   cmake -S engine -B build-engine -DSHUYI_CUDA=ON
#   cmake -S ui     -B build-ui
#
# and they never link each other: the UI talks to the engine over a pipe.
cmake_minimum_required(VERSION 3.20)
project(shuyi LANGUAGES CXX)

option(SHUYI_CUDA "Build the CUDA backend of the engine" OFF)
option(SHUYI_BUILD_ENGINE "Build shuyi_engine" ON)
# The front end is a Win32 program and refuses to configure anywhere else, so the
# default follows the platform: a Linux server can run plain `cmake -S . -B build`
# for the engine and tools instead of hitting a configure error.
if(WIN32)
    set(SHUYI_UI_DEFAULT ON)
else()
    set(SHUYI_UI_DEFAULT OFF)
endif()
option(SHUYI_BUILD_UI "Build the Win32 front end" ${SHUYI_UI_DEFAULT})
option(SHUYI_BUILD_TOOLS "Build developer tools (not part of any release)" OFF)

enable_testing()

# Both products land in the build root, side by side, exactly as they would in a
# downloaded package: the UI finds its engine next to itself.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(core)
if(SHUYI_BUILD_ENGINE)
    add_subdirectory(engine)
endif()
if(SHUYI_BUILD_UI)
    add_subdirectory(ui)
endif()
if(SHUYI_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# The training tools have no compiled code, but their arithmetic decides whether a
# rented ten-hour window trains a model or a few hundred games.  It is tested like
# code: the sizing maths and the deadline rule run in a second, no GPU involved.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME shuyi_training_budget
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/training/tests/test_budget.py)
    add_test(NAME shuyi_training_evaluation
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/training/tests/test_evaluation.py)
endif()

# -------------------------------------------------------------------- install --
# Each product installs on its own, into its own prefix:
#   cmake --install build --component engine --prefix dist/engine
#   cmake --install build --component ui     --prefix dist/ui
