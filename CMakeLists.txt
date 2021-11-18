cmake_minimum_required(VERSION 3.14)
project(dwarf_bench)

find_package(OpenCL REQUIRED)
find_package(oclhelpers 0.1.3 REQUIRED)
find_package(CUDA)

option(ENABLE_DPCPP ON)
option(ENABLE_CUDA OFF)
option(EXPLICIT_DPL OFF)

if(ENABLE_DPCPP)
  if(EXPLICIT_DPL)
    find_package(oneDPL REQUIRED)
    find_package(TBB REQUIRED)
  endif()
  add_definitions(-DDPCPP_ENABLED)  
endif()

if(CUDA_FOUND AND ENABLE_CUDA)
  message(STATUS "Using ${CUDA_VERSION} headers")
  if (${CUDA_VERSION_MAJOR} GREATER_EQUAL 11)
    add_definitions(-DCUDA_OCL_API_CHANGED)
  endif()
  add_definitions(-DCUDA_ENABLED)
endif()

find_package(Boost 1.61 COMPONENTS program_options filesystem REQUIRED)

list(APPEND CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/cmake
)

include(KernelFunctions)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(common)
add_subdirectory(scan)
add_subdirectory(constant)
add_subdirectory(radix)
add_subdirectory(reduce)
add_subdirectory(hash)
add_subdirectory(join)
add_subdirectory(probe)
add_subdirectory(groupby)

set(bench_libs 
  common
  standalone_scan
  constant
)

if(ENABLE_DPCPP)
  add_subdirectory(common/dpcpp)
  
  list(APPEND bench_libs
    dpcpp_constant
    scan
    radix
    reduce
    hash_build
    slab_hash_build
    slab_join
    slab_probe
    nested_loop_join
    join_helpers_lib
    join
    groupby
  )
  if(ENABLE_CUDA)
  list(APPEND bench_libs
      dpcpp_constant_cuda
      scan_cuda
      radix_cuda
  )
  endif()
endif()

set(${CMAKE_CXX_FLAGS_DEBUG} "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(bench_sources bench.cpp register_dwarfs.cpp register_dwarfs.hpp)
add_executable(${PROJECT_NAME} ${bench_sources})
target_link_libraries(${PROJECT_NAME} PRIVATE Boost::program_options ${bench_libs})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR})

add_kernel(vadd)
add_executable(simple simple.cpp)
target_link_libraries(simple PRIVATE common OpenCL::OpenCL oclhelpers::oclhelpers)
target_include_directories(simple PRIVATE ${PROJECT_SOURCE_DIR})

option(ENABLE_TESTS ON)
if(ENABLE_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  
  add_subdirectory(tests)
endif()
