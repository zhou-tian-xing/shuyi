# 数弈 engine: the headless NN-MCTS engine and its line protocol.
#
# This project is one product.  It links `shuyi_core` (the rules) and nothing
# from the UI, and it is built and shipped on its own:
#
#   cmake -S engine -B build-engine -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-engine
#
# A UI from another project — or another language — can drive the resulting
# shuyi_engine.exe as long as it speaks protocol/ENGINE_PROTOCOL.md.
cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(shuyi_engine_project LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SHUYI_CUDA "Build the CUDA backend" OFF)
option(SHUYI_ENGINE_EXECUTABLE "Build the shuyi_engine executable" ON)
option(SHUYI_ENGINE_TESTS "Build and register the C++ tests" ON)
# Optional -march=native.  Measured on an Ice Lake laptop it is slightly SLOWER
# for these small convolutions (4.3 ms vs 3.7 ms per b4c32 evaluation, probably
# AVX-512 downclocking), so it is off by default.
option(SHUYI_NATIVE_SIMD "Compile for the host CPU (-march=native / /arch:AVX2)" OFF)

set(SHUYI_ENGINE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(NOT TARGET shuyi_core)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../core ${CMAKE_BINARY_DIR}/core)
endif()

if(MSVC)
    add_compile_options(/utf-8)
else()
    add_compile_options(-finput-charset=UTF-8)
endif()
if(SHUYI_NATIVE_SIMD)
    if(MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-march=native)
    endif()
endif()

add_library(shuyi_search STATIC
    src/ai.cpp
    src/ai_blocking.cpp
    src/nn.cpp
    src/mcts.cpp
    src/engine.cpp
    src/tt.cpp
)
target_link_libraries(shuyi_search PUBLIC shuyi_core)
target_include_directories(shuyi_search PUBLIC ${SHUYI_ENGINE_INCLUDE})
if(MSVC)
    target_compile_options(shuyi_search PRIVATE /W4)
else()
    target_compile_options(shuyi_search PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Network inference is parallel over output channels; without OpenMP a CPU
# forward uses one core.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(shuyi_search PUBLIC OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found: CPU network inference stays single-threaded")
endif()

if(SHUYI_ENGINE_EXECUTABLE)
    add_executable(shuyi_engine
        src/main.cpp
        src/protocol.cpp
        src/protocol_json.cpp
    )
    target_link_libraries(shuyi_engine PRIVATE shuyi_search)
    if(MSVC)
        target_compile_options(shuyi_engine PRIVATE /W4)
    else()
        target_compile_options(shuyi_engine PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(SHUYI_CUDA)
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
                -I${CMAKE_CURRENT_SOURCE_DIR}/../core/include
                -c ${CMAKE_CURRENT_SOURCE_DIR}/src/nn_cuda.cu
                -o ${SHUYI_CUDA_OBJ}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/nn_cuda.cu
                ${SHUYI_ENGINE_INCLUDE}/shuyi/nn_cuda.h
                ${SHUYI_ENGINE_INCLUDE}/shuyi/nn.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../core/include/shuyi/position.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../core/include/shuyi/rules.h
        VERBATIM)
    add_library(shuyi_cuda STATIC ${SHUYI_CUDA_OBJ})
    target_link_libraries(shuyi_cuda PUBLIC shuyi_search CUDA::cudart CUDA::cublas)
    if(TARGET shuyi_engine)
        target_link_libraries(shuyi_engine PRIVATE shuyi_cuda)
        target_compile_definitions(shuyi_engine PRIVATE SHUYI_CUDA_ENGINE=1)
    endif()
endif()

if(SHUYI_ENGINE_TESTS)
    enable_testing()
    add_executable(shuyi_tests
        tests/test_position.cpp
        tests/test_rules.cpp
        tests/test_komi.cpp
        tests/test_engine.cpp
    )
    target_link_libraries(shuyi_tests PRIVATE shuyi_search)
    # Tests read fixtures (data/score_witnesses.txt, models/shuyi_resnet_v1.bin)
    # from the repository root.
    target_compile_definitions(shuyi_tests PRIVATE
        SHUYI_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    add_test(NAME shuyi_tests COMMAND shuyi_tests)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND AND TARGET shuyi_engine)
        # Protocol conformance: drives any engine binary, so it also checks an
        # engine built by someone else against protocol/ENGINE_PROTOCOL.md.
        add_test(NAME shuyi_protocol
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../protocol/tests/protocol_smoke.py
                    $<TARGET_FILE:shuyi_engine>)
        add_test(NAME shuyi_visit_budget
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../protocol/tests/visit_budget_smoke.py
                    $<TARGET_FILE:shuyi_engine>
                    ${CMAKE_CURRENT_SOURCE_DIR}/../models/shuyi_resnet_v1.bin)
        set_tests_properties(shuyi_protocol shuyi_visit_budget PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
    endif()
endif()

if(TARGET shuyi_engine)
    install(TARGETS shuyi_engine RUNTIME DESTINATION . COMPONENT engine)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../protocol/ENGINE_PROTOCOL.md
            DESTINATION . COMPONENT engine)
endif()
