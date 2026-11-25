# 数弈 UI: the native Win32 front end.
#
# One product, built on its own:
#
#   cmake -S ui -B build-ui -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-ui
#
# It links `shuyi_core` (the rules it is the authority for) and NOTHING from the
# engine: no search, no network, no CUDA.  All computation is requested from
# whatever engine binary the user points it at, over protocol/ENGINE_PROTOCOL.md.
cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(shuyi_ui_project LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT WIN32)
    message(FATAL_ERROR "The 数弈 UI is a native Win32 program; build it on Windows.")
endif()
if(NOT TARGET shuyi_core)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../core ${CMAKE_BINARY_DIR}/core)
endif()

if(MSVC)
    add_compile_options(/utf-8)
else()
    add_compile_options(-finput-charset=UTF-8)
endif()

add_executable(shuyi WIN32
    src/ui_win32.cpp
    src/engine_client.cpp
    src/engine_protocol.cpp
    src/json.cpp
    src/discovery.cpp
)
target_include_directories(shuyi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(shuyi PRIVATE shuyi_core comctl32 gdi32 user32)
if(MSVC)
    target_compile_options(shuyi PRIVATE /W4)
else()
    target_compile_options(shuyi PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(MINGW)
    target_link_options(shuyi PRIVATE -municode -static)
elseif(MSVC)
    target_link_options(shuyi PRIVATE /ENTRY:wWinMainCRTStartup)
endif()

add_custom_command(TARGET shuyi POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/../protocol/ENGINE_PROTOCOL.md
        $<TARGET_FILE_DIR:shuyi>/ENGINE_PROTOCOL.md)

install(TARGETS shuyi RUNTIME DESTINATION . COMPONENT ui)
install(FILES README.md ${CMAKE_CURRENT_SOURCE_DIR}/../protocol/ENGINE_PROTOCOL.md
        DESTINATION . COMPONENT ui)

enable_testing()
# The parser that turns the engine's JSON into what the window draws.  It is a
# separate binary because it needs no window: a broken parser must be caught
# before anyone sees an empty board.
add_executable(shuyi_ui_tests
    tests/test_engine_view.cpp
    src/engine_protocol.cpp
    src/json.cpp
)
target_include_directories(shuyi_ui_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(shuyi_ui_tests PRIVATE shuyi_core)
if(MSVC)
    target_compile_options(shuyi_ui_tests PRIVATE /utf-8 /W4)
else()
    target_compile_options(shuyi_ui_tests PRIVATE -finput-charset=UTF-8 -Wall -Wextra -Wpedantic)
endif()
add_test(NAME shuyi_ui_view COMMAND shuyi_ui_tests)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    # UI tests use the standalone binary, without linking the engine.
    add_test(NAME shuyi_ui_menu
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/ui_menu_smoke.py
                $<TARGET_FILE:shuyi>)
    add_test(NAME shuyi_engine_choice
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/engine_choice_smoke.py
                $<TARGET_FILE:shuyi>)
    add_test(NAME shuyi_ui_a_value
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/a_value_smoke.py
                $<TARGET_FILE:shuyi>)
    set_tests_properties(shuyi_ui_menu shuyi_engine_choice shuyi_ui_a_value PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()
