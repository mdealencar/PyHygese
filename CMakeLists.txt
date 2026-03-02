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
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  hgscvrp
  GIT_REPOSITORY https://github.com/vidalt/HGS-CVRP.git
  GIT_TAG 1a927955cd2861a29d978f0d359d6e647db9319c
)
FetchContent_MakeAvailable(hgscvrp)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

set(CYTHON_FILE ${CMAKE_SOURCE_DIR}/hybgensea/_hybgensea.pyx)
set(CYTHON_CPP_FILE ${CMAKE_CURRENT_BINARY_DIR}/_hybgensea.cpp)

add_custom_command(
    OUTPUT ${CYTHON_CPP_FILE}
    COMMAND ${Python_EXECUTABLE} -m cython --cplus -3 ${CYTHON_FILE} -o ${CYTHON_CPP_FILE}
    DEPENDS ${CYTHON_FILE}
    VERBATIM
)

Python_add_library(_hybgensea MODULE ${CYTHON_CPP_FILE} WITH_SOABI)
target_include_directories(_hybgensea PRIVATE ${Python_NumPy_INCLUDE_DIRS})
target_link_libraries(_hybgensea PRIVATE lib)

if(APPLE)
    set_target_properties(_hybgensea PROPERTIES INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set_target_properties(_hybgensea PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

install(TARGETS _hybgensea LIBRARY DESTINATION hybgensea)

if(CMAKE_HOST_WIN32)
  install(TARGETS lib RUNTIME DESTINATION hybgensea)
else()
  install(TARGETS lib LIBRARY DESTINATION hybgensea)
endif()
