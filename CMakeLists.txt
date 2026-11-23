cmake_minimum_required(VERSION 3.20)
project(shuyi LANGUAGES CXX)

# Layout (see docs/MAINTAINERS.zh-CN.md):
#   engine/       the rules + search + network library, and the line-protocol engine
#   ui/           the native Win32 front end and the process client that talks to the engine
#   experimental/ self-play, training, and one-off research tools
#   data/, models/, runs/  shared fixtures, runtime assets and run outputs

# The UI and the engine are separate programs that speak one line protocol, and
# they may come from different projects entirely.  Each is built and packaged on
# its own:
#   cmake -B build-engine -DSHUYI_BUILD_UI=OFF
#   cmake -B build-ui     -DSHUYI_BUILD_ENGINE=OFF
#   cmake --install build-ui --component ui --prefix dist/ui
option(SHUYI_CUDA "Build the CUDA backend and the batched self-play tool" OFF)
option(SHUYI_BUILD_UI "Build the Win32 front end" ON)
option(SHUYI_BUILD_ENGINE "Build the line-protocol engine" ON)
option(SHUYI_BUILD_EXPERIMENTAL "Build self-play / research tools and tests" ON)
# The UI talks to the engine over the protocol only.  Linking the search into the
# window (a built-in fallback AI) is opt-in, because it defeats the split: the UI
# package would then contain an engine.
option(SHUYI_UI_BUILTIN_AI "Link the search into the UI as a fallback player" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8)
else()
    add_compile_options(-finput-charset=UTF-8)
endif()

# Optional -march=native.  Measured on an Ice Lake laptop it is slightly SLOWER
# for these small convolutions (4.3 ms vs 3.7 ms per b4c32 evaluation, probably
# AVX-512 downclocking), so it is off by default.  Enable it per machine only if
# a benchmark says it helps:
#   cmake -B build -DSHUYI_NATIVE_SIMD=ON
option(SHUYI_NATIVE_SIMD "Compile for the host CPU (-march=native / /arch:AVX2)" OFF)
if(SHUYI_NATIVE_SIMD)
    if(MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-march=native)
    endif()
endif()

set(SHUYI_ENGINE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/engine/include)
set(SHUYI_UI_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ui/include)

# --------------------------------------------------------------------- engine --
# The rules (board, scoring, move legality) are what a *front end* needs to draw
# and validate; the search and the network are what only an *engine* needs.
# Keeping them in two libraries is what lets the UI package be just a UI.
add_library(shuyi_rules STATIC
    engine/src/position.cpp
    engine/src/tt.cpp
)
target_include_directories(shuyi_rules PUBLIC ${SHUYI_ENGINE_INCLUDE})
if(MSVC)
    target_compile_options(shuyi_rules PRIVATE /W4)
else()
    target_compile_options(shuyi_rules PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(shuyi_search STATIC
    engine/src/ai.cpp
    engine/src/ai_blocking.cpp
    engine/src/nn.cpp
    engine/src/mcts.cpp
    engine/src/engine.cpp
)
target_link_libraries(shuyi_search PUBLIC shuyi_rules)
if(MSVC)
    target_compile_options(shuyi_search PRIVATE /W4)
else()
    target_compile_options(shuyi_search PRIVATE -Wall -Wextra -Wpedantic)
endif()

# The CPU network forward is parallel over output channels; without OpenMP the
# UI's MCTS+NN gets one core's worth of inference.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(shuyi_search PUBLIC OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found: CPU network inference stays single-threaded")
endif()

if(SHUYI_BUILD_ENGINE)
add_executable(shuyi_engine engine/src/engine_main.cpp)
target_link_libraries(shuyi_engine PRIVATE shuyi_search)
if(MSVC)
    target_compile_options(shuyi_engine PRIVATE /W4)
else()
    target_compile_options(shuyi_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()
endif()  # SHUYI_BUILD_ENGINE

add_executable(shuyi_tests
    engine/tests/test_position.cpp
    engine/tests/test_rules.cpp
    engine/tests/test_komi.cpp
    engine/tests/test_engine.cpp
)
target_link_libraries(shuyi_tests PRIVATE shuyi_search)
# Tests read fixtures (data/score_witnesses.txt, models/shuyi_resnet_v1.bin) from
# the source tree, so they get the root, not the build directory.
target_compile_definitions(shuyi_tests PRIVATE
    SHUYI_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

# ------------------------------------------------------------------------- ui --
if(WIN32 AND SHUYI_BUILD_UI)
    add_executable(shuyi WIN32 ui/src/ui_win32.cpp ui/src/engine_client.cpp)
    target_include_directories(shuyi PRIVATE ${SHUYI_UI_INCLUDE})
    # Rules + the protocol client.  NOT the search: the window runs no AI of its
    # own unless SHUYI_UI_BUILTIN_AI is switched on.
    target_link_libraries(shuyi PRIVATE shuyi_rules comctl32 gdi32 user32)
    if(SHUYI_UI_BUILTIN_AI)
        target_link_libraries(shuyi PRIVATE shuyi_search)
        target_compile_definitions(shuyi PRIVATE SHUYI_UI_BUILTIN_AI=1)
    endif()
    if(MSVC)
        target_compile_options(shuyi PRIVATE /W4)
    else()
        target_compile_options(shuyi PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    if(MINGW)
        target_link_options(shuyi PRIVATE -municode -static-libgcc -static-libstdc++)
    elseif(MSVC)
        target_link_options(shuyi PRIVATE /ENTRY:wWinMainCRTStartup)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models")
        add_custom_command(TARGET shuyi POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/models" "$<TARGET_FILE_DIR:shuyi>/models")
    endif()
endif()

# --------------------------------------------------------------- experimental --
if(SHUYI_BUILD_EXPERIMENTAL)
add_executable(shuyi_explore experimental/research/komi_explore.cpp)
target_link_libraries(shuyi_explore PRIVATE shuyi_search)

add_executable(shuyi_a9_lemma experimental/research/a9_lemma.cpp)

add_executable(shuyi_nnbench experimental/research/nnbench.cpp)
target_link_libraries(shuyi_nnbench PRIVATE shuyi_search)

# Two manual tools for the endgame solver.  proof_check replays random endgames
# and re-solves every position the engine calls "proven" with an independent
# exact search; endgame_probe prints the exact value of every move next to what
# MCTS reports for one position.
add_executable(shuyi_proof_check experimental/research/proof_check.cpp)
target_link_libraries(shuyi_proof_check PRIVATE shuyi_search)
add_executable(shuyi_endgame_probe experimental/research/endgame_probe.cpp)
target_link_libraries(shuyi_endgame_probe PRIVATE shuyi_search)

add_executable(shuyi_selfplay experimental/selfplay/selfplay.cpp)
target_link_libraries(shuyi_selfplay PRIVATE shuyi_search)
if(MSVC)
    target_compile_options(shuyi_selfplay PRIVATE /W4)
else()
    target_compile_options(shuyi_selfplay PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(shuyi_tournament experimental/selfplay/tournament.cpp)
target_link_libraries(shuyi_tournament PRIVATE shuyi_search)
if(MSVC)
    target_compile_options(shuyi_tournament PRIVATE /W4)
else()
    target_compile_options(shuyi_tournament PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(shuyi_match experimental/selfplay/match.cpp)
target_link_libraries(shuyi_match PRIVATE shuyi_search)
if(MSVC)
    target_compile_options(shuyi_match PRIVATE /W4)
else()
    target_compile_options(shuyi_match PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(SHUYI_CUDA)
    # Compile the .cu through a custom command: CMake's CUDA language support
    # generates an nvcc command line that 12.4 rejects on Windows/MSVC.
    find_package(CUDAToolkit REQUIRED)
    # "native" builds only for the GPU that is actually installed, which is what
    # we want on a rented box: nvcc would otherwise spend minutes per arch.
    set(SHUYI_CUDA_ARCH "native" CACHE STRING
        "nvcc -arch value (native, sm_86, or a ;-separated arch list)")
    set(SHUYI_NVCC_ARCH_FLAGS "")
    if(SHUYI_CUDA_ARCH MATCHES ";")
        foreach(arch ${SHUYI_CUDA_ARCH})
            list(APPEND SHUYI_NVCC_ARCH_FLAGS
                 "--generate-code=arch=compute_${arch},code=sm_${arch}")
        endforeach()
    else()
        list(APPEND SHUYI_NVCC_ARCH_FLAGS "-arch=${SHUYI_CUDA_ARCH}")
    endif()
    set(SHUYI_NVCC_FLAGS "")
    if(MSVC)
        list(APPEND SHUYI_NVCC_FLAGS -Xcompiler=/utf-8 -Xcompiler=/EHsc -Xcompiler=/MD)
    else()
        list(APPEND SHUYI_NVCC_FLAGS -Xcompiler=-fPIC)
    endif()
    set(SHUYI_CUDA_OBJ ${CMAKE_BINARY_DIR}/nn_cuda${CMAKE_CXX_OUTPUT_EXTENSION})
    add_custom_command(
        OUTPUT ${SHUYI_CUDA_OBJ}
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
                -std=c++17 -O3
                ${SHUYI_NVCC_FLAGS}
                ${SHUYI_NVCC_ARCH_FLAGS}
                -I${SHUYI_ENGINE_INCLUDE}
                -c ${CMAKE_CURRENT_SOURCE_DIR}/engine/src/nn_cuda.cu
                -o ${SHUYI_CUDA_OBJ}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/engine/src/nn_cuda.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/engine/include/shuyi/nn_cuda.h
                ${CMAKE_CURRENT_SOURCE_DIR}/engine/include/shuyi/nn.h
                ${CMAKE_CURRENT_SOURCE_DIR}/engine/include/shuyi/position.h
                ${CMAKE_CURRENT_SOURCE_DIR}/engine/include/shuyi/rules.h
        VERBATIM)
    add_library(shuyi_cuda STATIC ${SHUYI_CUDA_OBJ})
    target_include_directories(shuyi_cuda PUBLIC ${SHUYI_ENGINE_INCLUDE})
    # Dynamic cudart on purpose: cuBLAS links the dynamic runtime itself, and
    # mixing a static cudart into the executable makes cublasSgemm fail with
    # CUBLAS_STATUS_EXECUTION_FAILED at the first call.
    target_link_libraries(shuyi_cuda PUBLIC shuyi_search CUDA::cudart CUDA::cublas)
    # The UI drives shuyi_engine, so the engine needs the CUDA forward too.
    if(TARGET shuyi_engine)
        target_link_libraries(shuyi_engine PRIVATE shuyi_cuda)
        target_compile_definitions(shuyi_engine PRIVATE SHUYI_CUDA_ENGINE=1)
    endif()

    add_executable(shuyi_selfplay_gpu experimental/selfplay/selfplay_gpu.cpp)
    target_link_libraries(shuyi_selfplay_gpu PRIVATE shuyi_cuda shuyi_search)
    # The CPU half of a self-play tick (descend/expand/back up) is parallel; without
    # OpenMP the GPU idles ~70% of the time waiting for one core.
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(shuyi_selfplay_gpu PRIVATE OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP not found: self-play stays single-threaded on the CPU side")
    endif()
endif()

endif()  # SHUYI_BUILD_EXPERIMENTAL

# ----------------------------------------------------------------------- tests --
enable_testing()
find_package(Python3 COMPONENTS Interpreter QUIET)
# The UI package can be tested on its own: these only need the window binary.
if(TARGET shuyi AND Python3_Interpreter_FOUND)
    add_test(NAME shuyi_ui_menu
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ui/tests/ui_menu_smoke.py
                $<TARGET_FILE:shuyi>)
    set_tests_properties(shuyi_ui_menu PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    # The UI/engine split means the window must spawn the engine the user chose,
    # and only fall back to discovery when nothing is configured.
    add_test(NAME shuyi_engine_choice
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ui/tests/engine_choice_smoke.py
                $<TARGET_FILE:shuyi>)
    set_tests_properties(shuyi_engine_choice PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(SHUYI_BUILD_EXPERIMENTAL)
    add_test(NAME shuyi_tests COMMAND shuyi_tests)
    add_test(NAME shuyi_match_smoke
        COMMAND shuyi_match --a-engine random --b-engine pruning --games 2 --time 2
                --opening 0 --seed 7 --out test_match.json --csv test_match.csv)
    add_test(NAME shuyi_tournament_smoke
        COMMAND shuyi_tournament --games 2 --time 2 --seed 7
                --engines random,pruning
                --out test_elo.md --csv test_elo.csv)
    if(WIN32)
        add_test(NAME shuyi_a9_lemma COMMAND shuyi_a9_lemma)
    endif()
    if(Python3_Interpreter_FOUND)
        add_test(NAME shuyi_protocol
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/engine/tests/protocol_smoke.py
                    $<TARGET_FILE:shuyi_engine>)
        add_test(NAME shuyi_selfplay_smoke
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/experimental/tests/selfplay_smoke.py
                    $<TARGET_FILE:shuyi_selfplay>)
        # A rating is about the model, so every engine must honour a visit budget.
        add_test(NAME shuyi_visit_budget
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/engine/tests/visit_budget_smoke.py
                    $<TARGET_FILE:shuyi_engine>
                    ${CMAKE_CURRENT_SOURCE_DIR}/models/shuyi_resnet_v1.bin)
        set_tests_properties(shuyi_visit_budget PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
endif()

# -------------------------------------------------------------------- install --
# Each side installs on its own, so a front end and an engine can ship, version
# and be downloaded separately:
#   cmake --install build --component ui    --prefix dist/ui
#   cmake --install build --component engine --prefix dist/engine
if(TARGET shuyi)
    install(TARGETS shuyi RUNTIME DESTINATION . COMPONENT ui)
    install(FILES README.md DESTINATION . COMPONENT ui)
endif()
if(TARGET shuyi_engine)
    install(TARGETS shuyi_engine RUNTIME DESTINATION . COMPONENT engine)
    install(FILES docs/ENGINE_PROTOCOL.md DESTINATION . COMPONENT engine)
endif()
if(SHUYI_BUILD_EXPERIMENTAL)
    install(TARGETS shuyi_selfplay shuyi_match shuyi_tournament shuyi_nnbench
            shuyi_explore RUNTIME DESTINATION . COMPONENT experimental)
endif()
