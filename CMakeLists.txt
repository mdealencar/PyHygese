cmake_minimum_required(VERSION 3.15)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR} CACHE PATH "install prefix" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  hgscvrp
  GIT_REPOSITORY https://github.com/mdealencar/HGS-CVRP.git
  GIT_TAG 0b4482169f2695b0d153c1da5b4342ef2d79b76e
)
FetchContent_MakeAvailable(hgscvrp)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
  ERROR_QUIET
  RESULT_VARIABLE nanobind_exit_code
)
if(NOT nanobind_exit_code EQUAL 0)
  unset(nanobind_ROOT)
endif()
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core hybgensea/_core.cpp)
target_include_directories(_core PRIVATE ${hgscvrp_SOURCE_DIR}/Program)
target_link_libraries(_core PRIVATE objlib)

install(TARGETS _core LIBRARY DESTINATION hybgensea)
