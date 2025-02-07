cmake_minimum_required(VERSION 3.15)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  hgscvrp
  GIT_REPOSITORY https://github.com/mdealencar/HGS-CVRP
  # GIT_TAG main
  GIT_TAG 9b5e7c612da09e94960f410b715a3beeb7f6f122 # main branch 2025-02-07
)
FetchContent_MakeAvailable(hgscvrp)

install(TARGETS ${SKBUILD_PROJECT_NAME} DESTINATION .)
