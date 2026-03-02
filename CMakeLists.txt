cmake_minimum_required(VERSION 3.15)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR} CACHE PATH "install prefix" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(WIN32)
    set(CMAKE_SHARED_LIBRARY_PREFIX_CXX "lib")
endif()

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  hgscvrp
  GIT_REPOSITORY https://github.com/vidalt/HGS-CVRP.git
  GIT_TAG 1a927955cd2861a29d978f0d359d6e647db9319c
)
FetchContent_MakeAvailable(hgscvrp)

FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(_core STABLE_ABI hybgensea/_core.cpp)
target_link_libraries(_core PRIVATE lib)

install(TARGETS _core LIBRARY DESTINATION hybgensea)
if(CMAKE_HOST_WIN32)
  install(TARGETS lib RUNTIME DESTINATION hybgensea)
else()
  install(TARGETS lib LIBRARY DESTINATION hybgensea)
endif()
