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
option(SHUYI_BUILD_UI "Build the Win32 front end" ON)
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

# -------------------------------------------------------------------- install --
# Each product installs on its own, into its own prefix:
#   cmake --install build --component engine --prefix dist/engine
#   cmake --install build --component ui     --prefix dist/ui
