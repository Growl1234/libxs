if(TARGET libxs::libxs)
  return()
endif()

get_filename_component(_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

find_library(LIBXS_LIBRARY NAMES xs HINTS "${_prefix}/lib" NO_DEFAULT_PATH)
find_path(LIBXS_INCLUDE_DIR NAMES libxs/libxs.h HINTS "${_prefix}/include" "${_prefix}" NO_DEFAULT_PATH)

if(LIBXS_LIBRARY AND LIBXS_INCLUDE_DIR)
  find_package(Threads QUIET)
  set(_incdirs "${LIBXS_INCLUDE_DIR}")
  if(EXISTS "${LIBXS_INCLUDE_DIR}/libxs/libxs.mod")
    list(APPEND _incdirs "${LIBXS_INCLUDE_DIR}/libxs")
  endif()
  add_library(libxs::libxs UNKNOWN IMPORTED)
  set_target_properties(libxs::libxs PROPERTIES
    IMPORTED_LOCATION "${LIBXS_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${_incdirs}")
  if(TARGET Threads::Threads)
    set_property(TARGET libxs::libxs APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES Threads::Threads)
  endif()
  unset(_incdirs)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libxs DEFAULT_MSG LIBXS_LIBRARY LIBXS_INCLUDE_DIR)
unset(_prefix)
