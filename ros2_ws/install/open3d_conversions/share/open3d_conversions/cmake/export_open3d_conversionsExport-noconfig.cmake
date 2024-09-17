#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "open3d_conversions::open3d_conversions" for configuration ""
set_property(TARGET open3d_conversions::open3d_conversions APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(open3d_conversions::open3d_conversions PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libopen3d_conversions.so"
  IMPORTED_SONAME_NOCONFIG "libopen3d_conversions.so"
  )

list(APPEND _cmake_import_check_targets open3d_conversions::open3d_conversions )
list(APPEND _cmake_import_check_files_for_open3d_conversions::open3d_conversions "${_IMPORT_PREFIX}/lib/libopen3d_conversions.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
