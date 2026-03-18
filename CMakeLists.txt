cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(libxs
  VERSION 0.0.0
  LANGUAGES C)

option(LIBXS_HEADER_ONLY "Header-only usage (no library target)" OFF)
option(LIBXS_SHARED      "Build shared library instead of static" OFF)

message(STATUS "******** Build Summary ********")
message(STATUS "  CMake version         : ${CMAKE_VERSION}")
message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  LIBXS_HEADER_ONLY     : ${LIBXS_HEADER_ONLY}")
message(STATUS "  LIBXS_SHARED          : ${LIBXS_SHARED}")

set(LIBXS_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(LIBXS_INCLUDE_DIR ${LIBXS_ROOT_DIR}/include)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(LIBXS_HEADER_ONLY)
  add_library(libxs INTERFACE)
  target_include_directories(libxs INTERFACE
    $<BUILD_INTERFACE:${LIBXS_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)
  add_library(libxs::libxs ALIAS libxs)
else()
  file(GLOB LIBXS_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS
    ${LIBXS_ROOT_DIR}/src/*.c)

  if(LIBXS_SHARED)
    add_library(libxs SHARED ${LIBXS_SRCS})
  else()
    add_library(libxs STATIC ${LIBXS_SRCS})
  endif()
  add_library(libxs::libxs ALIAS libxs)

  target_compile_definitions(libxs PRIVATE LIBXS_BUILD=1)

  target_include_directories(libxs
    PUBLIC
      $<BUILD_INTERFACE:${LIBXS_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${LIBXS_ROOT_DIR}/src)

  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  target_link_libraries(libxs PUBLIC Threads::Threads)
  target_link_libraries(libxs PUBLIC ${CMAKE_DL_LIBS})

  include(CheckLibraryExists)
  check_library_exists(m sqrt "" LIBXS_HAS_LIBM)
  if(LIBXS_HAS_LIBM)
    target_link_libraries(libxs PUBLIC m)
  endif()
  check_library_exists(rt clock_gettime "" LIBXS_HAS_LIBRT)
  if(LIBXS_HAS_LIBRT)
    target_link_libraries(libxs PUBLIC rt)
  endif()
endif()
