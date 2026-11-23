# 数弈 developer tools.
#
# Never part of a release and never built by the umbrella by default.  They link
# the search library directly, which is exactly what a product must not do:
# these are the programs that inspect, benchmark and train the engine.
#
#   cmake -S tools -B build-tools -DCMAKE_BUILD_TYPE=Release
cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(shuyi_tools LANGUAGES CXX)
endif()

if(NOT TARGET shuyi_search)
    set(SHUYI_ENGINE_EXECUTABLE OFF CACHE BOOL "" FORCE)
    set(SHUYI_ENGINE_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../engine ${CMAKE_BINARY_DIR}/engine)
endif()

macro(shuyi_tool name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE shuyi_search)
    if(MSVC)
        # The sources are UTF-8 and the report generators contain Chinese
        # literals; without /utf-8 MSVC reads them as the local code page and
        # eats the closing quote (C2001/C3688).
        target_compile_options(${name} PRIVATE /utf-8 /W4)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endmacro()

shuyi_tool(shuyi_explore research/komi_explore.cpp)
shuyi_tool(shuyi_a9_lemma research/a9_lemma.cpp)
shuyi_tool(shuyi_nnbench research/nnbench.cpp)
shuyi_tool(shuyi_proof_check research/proof_check.cpp)
shuyi_tool(shuyi_endgame_probe research/endgame_probe.cpp)
shuyi_tool(shuyi_selfplay selfplay/selfplay.cpp)
shuyi_tool(shuyi_tournament selfplay/tournament.cpp)
shuyi_tool(shuyi_match selfplay/match.cpp)

if(SHUYI_CUDA AND TARGET shuyi_cuda)
    add_executable(shuyi_selfplay_gpu selfplay/selfplay_gpu.cpp)
    target_link_libraries(shuyi_selfplay_gpu PRIVATE shuyi_cuda shuyi_search)
    if(MSVC)
        target_compile_options(shuyi_selfplay_gpu PRIVATE /utf-8)
    endif()
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(shuyi_selfplay_gpu PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

enable_testing()
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME shuyi_match_smoke
        COMMAND shuyi_match --a-engine random --b-engine pruning --games 2 --time 2
                --opening 0 --seed 7 --out test_match.json --csv test_match.csv)
    add_test(NAME shuyi_tournament_smoke
        COMMAND shuyi_tournament --games 2 --time 2 --seed 7
                --engines random,pruning
                --out test_elo.md --csv test_elo.csv)
    add_test(NAME shuyi_selfplay_smoke
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/selfplay_smoke.py
                $<TARGET_FILE:shuyi_selfplay>)
endif()
if(WIN32)
    add_test(NAME shuyi_a9_lemma COMMAND shuyi_a9_lemma)
endif()
