# python-pcl_skbuild top-level build.
# Structure follows sirokujira/cython-scikit-build-template.
cmake_minimum_required(VERSION 3.12.0)

project(python_pcl)

# scikit-build CMake modules
find_package(PythonInterp)
find_package(PythonLibs)
find_package(PythonExtensions)
find_package(Cython)
find_package(NumPy)

# PCL_ROOT can point at a custom PCL install (mirrors the template's
# EXTERNAL_MODULE_PATH handling).
if(DEFINED ENV{PCL_ROOT})
    list(APPEND CMAKE_MODULE_PATH "$ENV{PCL_ROOT}/cmake")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{PCL_ROOT}")
endif()

# https://scikit-build.readthedocs.io/en/latest/generators.html#macosx
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13")
endif(APPLE)

find_package(PCL REQUIRED COMPONENTS common)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src ${PCL_INCLUDE_DIRS})
if(NumPy_FOUND)
    include_directories(${NumPy_INCLUDE_DIRS})
endif()
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_subdirectory(src/pcl)
