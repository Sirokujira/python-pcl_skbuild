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
#
# TO_CMAKE_PATH is load-bearing: on Windows the variable arrives with
# backslashes, and a backslashed entry in CMAKE_MODULE_PATH blows up the
# first try_compile that forwards the list into a scratch project —
# "Invalid character escape '\U'" from C:\Users\... — deep inside
# find_package(PCL)'s dependency probing rather than anywhere near the
# actual mistake.
if(DEFINED ENV{PCL_ROOT})
    file(TO_CMAKE_PATH "$ENV{PCL_ROOT}" _pcl_root)
    list(APPEND CMAKE_MODULE_PATH "${_pcl_root}/cmake")
    list(APPEND CMAKE_PREFIX_PATH "${_pcl_root}")
endif()

# https://scikit-build.readthedocs.io/en/latest/generators.html#macosx
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13")
endif(APPLE)

find_package(PCL REQUIRED COMPONENTS
    common io filters kdtree octree search segmentation features surface
    registration keypoints recognition tracking people stereo ml)

# Optional: pcl_outofcore. PCL builds it only together with its VTK-based
# visualization module, and its templates include VTK headers. Asking
# find_package(PCL) for it would make VTK REQUIRED through that module,
# so look for the library and header directly, plus the VTK targets;
# _outofcore is built only when all three are there (src/pcl/CMakeLists.txt).
find_library(PCL_OUTOFCORE_LIBRARY
    NAMES pcl_outofcore${PCL_RELEASE_SUFFIX} pcl_outofcore
    HINTS ${PCL_LIBRARY_DIRS} NO_DEFAULT_PATH)
find_path(PCL_OUTOFCORE_HEADER pcl/outofcore/outofcore.h
    HINTS ${PCL_INCLUDE_DIRS} NO_DEFAULT_PATH)
# VTK's headers reach us through its imported targets (PCL links them
# when it was built with VTK), not through PCL_INCLUDE_DIRS.
if(PCL_OUTOFCORE_LIBRARY AND PCL_OUTOFCORE_HEADER AND TARGET VTK::CommonCore)
    set(PCL_HAVE_OUTOFCORE ON)
    message(STATUS "pcl_outofcore found: building pcl._outofcore")
else()
    set(PCL_HAVE_OUTOFCORE OFF)
    message(STATUS "pcl_outofcore (or VTK headers) not found: no pcl._outofcore")
endif()

# Optional: PCL's VTK mesh code -- smoothing, subdivision and decimation
# in pcl_surface, STL / VTK / PLY / OBJ mesh files in pcl_io. PCL
# compiles both, and installs their headers, only when it was built with
# VTK; the shim also calls VTK's filters itself
# (compat/vtk_mesh_args.h), so their targets must be there too.
find_path(PCL_SURFACE_VTK_HEADER
    pcl/surface/vtk_smoothing/vtk_mesh_smoothing_laplacian.h
    HINTS ${PCL_INCLUDE_DIRS} NO_DEFAULT_PATH)
find_path(PCL_IO_VTK_HEADER pcl/io/vtk_lib_io.h
    HINTS ${PCL_INCLUDE_DIRS} NO_DEFAULT_PATH)
if(PCL_SURFACE_VTK_HEADER AND PCL_IO_VTK_HEADER
        AND TARGET VTK::FiltersModeling AND TARGET VTK::FiltersCore
        AND TARGET VTK::CommonExecutionModel)
    set(PCL_HAVE_VTK_MESH ON)
    message(STATUS "PCL has its VTK mesh code: building pcl._vtk_mesh")
else()
    set(PCL_HAVE_VTK_MESH OFF)
    message(STATUS "PCL built without VTK: no pcl._vtk_mesh")
endif()

# Supported range: PCL 1.11+. 1.11 is where boost::shared_ptr became
# std::shared_ptr across the API; the generated pxd declare std::shared_ptr
# unconditionally, so an older PCL fails deep inside the C++ compile with
# unreadable template errors. Fail here instead, with a message that says
# what to do. See .claude/rules/pipeline.md for the version policy.
if(PCL_VERSION VERSION_LESS "1.11")
    message(FATAL_ERROR
        "PCL ${PCL_VERSION} found, but this package requires PCL 1.11 or newer "
        "(1.11 switched boost::shared_ptr -> std::shared_ptr). Point PCL_ROOT "
        "at a newer install, or use sirokujira/python-pcl for PCL 1.7/1.8.")
endif()

# PCL passes OpenMP on as the OpenMP::OpenMP_CXX target, whose compile
# options (-fopenmp, /openmp) reach every module through the PCL targets'
# usage requirements. Make them conditional on the CONSUMING target, so a
# module can opt out with the PCL_WITHOUT_OPENMP property -- needed for
# header-only PCL code whose OpenMP pragmas do not compile on current
# compilers (see src/pcl/_keypoints_serial.pyx).
if(TARGET OpenMP::OpenMP_CXX)
    get_target_property(_pcl_omp_options OpenMP::OpenMP_CXX INTERFACE_COMPILE_OPTIONS)
    if(_pcl_omp_options)
        set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS
            "$<$<NOT:$<BOOL:$<TARGET_PROPERTY:PCL_WITHOUT_OPENMP>>>:${_pcl_omp_options}>")
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src ${PCL_INCLUDE_DIRS})
if(NumPy_FOUND)
    include_directories(${NumPy_INCLUDE_DIRS})
endif()
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_subdirectory(src/pcl)
