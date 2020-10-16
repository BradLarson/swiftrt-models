cmake_minimum_required(VERSION 3.17)
project(Models
  LANGUAGES C Swift)

option(SWIFTRT_ENABLE_CUDA "Enable CUDA Support" NO)
if(SWIFTRT_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)
file(MAKE_DIRECTORY ${CMAKE_Swift_MODULE_DIRECTORY})

include(CTest)
include(ExternalProject)
include(SwiftSupport)

# Dependencies
ExternalProject_Add(swift-argument-parser
  GIT_REPOSITORY git://github.com/apple/swift-argument-parser.git
  GIT_TAG main
  CMAKE_ARGS
    -DBUILD_SHARED_LIBS=YES
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DCMAKE_Swift_COMPILER=${CMAKE_Swift_COMPILER}
    -DCMAKE_Swift_FLAGS=${CMAKE_Swift_FLAGS}
    -DBUILD_EXAMPLES=NO
    -DBUILD_TESTING=NO
  INSTALL_COMMAND
    ""
  BUILD_BYPRODUCTS
    <BINARY_DIR>/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ArgumentParser${CMAKE_SHARED_LIBRARY_SUFFIX}
  STEP_TARGETS install)
ExternalProject_Get_Property(swift-argument-parser BINARY_DIR)

add_library(ArgumentParser SHARED IMPORTED)
set_target_properties(ArgumentParser PROPERTIES
  IMPORTED_LOCATION ${BINARY_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ArgumentParser${CMAKE_SHARED_LIBRARY_SUFFIX}
  INTERFACE_INCLUDE_DIRECTORIES ${BINARY_DIR}/swift)
add_dependencies(ArgumentParser swift-argument-parser-install)

file(MAKE_DIRECTORY ${BINARY_DIR}/swift)

ExternalProject_Add(swift-rt
  GIT_REPOSITORY git://github.com/ewconnell/swiftrt.git
  GIT_TAG master
  CMAKE_ARGS
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DBUILD_SHARED_LIBS=YES
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
    -DCMAKE_Swift_COMPILER=${CMAKE_Swift_COMPILER}
    -DCMAKE_Swift_FLAGS=${CMAKE_Swift_FLAGS}
    -DBUILD_TESTING=NO
    -DSWIFTRT_ENABLE_CUDA=${SWIFTRT_ENABLE_CUDA}
    -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
  INSTALL_COMMAND
    ""
  BUILD_BYPRODUCTS
    <BINARY_DIR>/lib/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftRT${CMAKE_SHARED_LIBRARY_SUFFIX}
    <BINARY_DIR>/lib/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftRTCuda${CMAKE_SHARED_LIBRARY_SUFFIX}
  STEP_TARGETS install)
ExternalProject_Get_Property(swift-rt BINARY_DIR)
ExternalProject_Get_Property(swift-rt SOURCE_DIR)

if(SWIFTRT_ENABLE_CUDA)
  add_library(CCUDA INTERFACE IMPORTED)
  set_target_properties(CCUDA PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${BINARY_DIR}/Sources/CCUDA;${CUDAToolkit_INCLUDE_DIRS}")

  add_library(SwiftRTCuda SHARED IMPORTED)
  set_target_properties(SwiftRTCuda PROPERTIES
    LANGUAGE CUDA
    IMPORTED_LOCATION ${BINARY_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftRTCuda${CMAKE_SHARED_LIBRARY_SUFFIX}
    INTERFACE_INCLUDE_DIRECTORIES ${SOURCE_DIR}/Modules/SwiftRTCuda
    INTERFACE_LINK_LIBRARIES CCUDA)
endif()

add_library(SwiftRT SHARED IMPORTED)
set_target_properties(SwiftRT PROPERTIES
  IMPORTED_LOCATION ${BINARY_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}SwiftRT${CMAKE_SHARED_LIBRARY_SUFFIX}
  INTERFACE_INCLUDE_DIRECTORIES ${BINARY_DIR}/swift
  INTERFACE_LINK_LIBRARIES $<$<BOOL:SWIFTRT_ENABLE_CUDA>:SwiftRTCuda>)
add_dependencies(SwiftRT swift-rt-install)

file(MAKE_DIRECTORY ${BINARY_DIR}/swift)
if(SWIFTRT_ENABLE_CUDA)
  file(MAKE_DIRECTORY ${SOURCE_DIR}/Modules/SwiftRTCuda)
  file(MAKE_DIRECTORY ${BINARY_DIR}/Sources/CCUDA)
endif()

function(import_module module_name build_dir build_target)
  add_library(${module_name} IMPORTED UNKNOWN)
  if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    set_target_properties(${module_name} PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${build_dir}/swift)
  else()
    set_target_properties(${module_name} PROPERTIES
      IMPORTED_LOCATION ${build_dir}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${module_name}${CMAKE_SHARED_LIBRARY_SUFFIX}
      INTERFACE_INCLUDE_DIRECTORIES ${build_dir}/swift)
  endif()
  add_dependencies(${module_name} ${build_target})
endfunction()

## swift-numerics
ExternalProject_Add(swift-numerics
  GIT_REPOSITORY
    git://github.com/apple/swift-numerics
  GIT_TAG
    master
  CMAKE_ARGS
    -D BUILD_SHARED_LIBS=YES
    -D BUILD_TESTING=NO
    -D CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -D CMAKE_Swift_COMPILER=${CMAKE_Swift_COMPILER}
    -D CMAKE_Swift_COMPILER_TARGET=${CMAKE_Swift_COMPILER_TARGET}
    -D CMAKE_Swift_FLAGS=${CMAKE_Swift_FLAGS}
  INSTALL_COMMAND
    ""
  BUILD_BYPRODUCTS
    <BINARY_DIR>/lib/${CMAKE_SHARED_LIBRARY_PREFIX}Numerics${CMAKE_SHARED_LIBRARY_SUFFIX}
  UPDATE_DISCONNECTED
    TRUE
  STEP_TARGETS
    build)
ExternalProject_Get_Property(swift-numerics BINARY_DIR)
ExternalProject_Get_Property(swift-numerics SOURCE_DIR)

file(MAKE_DIRECTORY ${BINARY_DIR}/swift)
file(MAKE_DIRECTORY ${SOURCE_DIR}/Sources/_NumericsShims/include)

import_module(Numerics ${BINARY_DIR} swift-numerics-build)
import_module(ComplexModule ${BINARY_DIR} swift-numerics-build)
import_module(RealModule ${BINARY_DIR} swift-numerics-build)

add_library(_NumericsShims IMPORTED INTERFACE)
set_target_properties(_NumericsShims PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${SOURCE_DIR}/Sources/_NumericsShims/include)
add_dependencies(_NumericsShims swift-numerics-build)

set_target_properties(Numerics PROPERTIES
  INTERFACE_LINK_DIRECTORIES ${BINARY_DIR}/lib
  INTERFACE_LINK_LIBRARIES _NumericsShims)

add_subdirectory(Support)
add_subdirectory(Examples)

