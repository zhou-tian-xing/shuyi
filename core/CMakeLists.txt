# 数弈 core: the rules, and nothing else.
#
# Board state, move legality, scoring and the symmetry transforms.  It knows
# nothing about search, networks, files or Windows, so it is the one piece that
# *both* products may link: the engine searches it, the UI renders and validates
# it.  Linking this is not linking the engine.
#
# Standalone build:
#   cmake -S core -B build-core
cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(shuyi_core LANGUAGES CXX)
endif()
if(TARGET shuyi_core)
    return()
endif()

add_library(shuyi_core STATIC src/position.cpp)
target_include_directories(shuyi_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(shuyi_core PUBLIC cxx_std_17)
if(MSVC)
    # The source is UTF-8 and contains Chinese literals; without /utf-8 MSVC
    # reads them in the ANSI code page and fails on the string suffixes.
    target_compile_options(shuyi_core PRIVATE /utf-8 /W4)
else()
    target_compile_options(shuyi_core PRIVATE -finput-charset=UTF-8 -Wall -Wextra -Wpedantic)
endif()
