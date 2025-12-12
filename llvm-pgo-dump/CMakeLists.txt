cmake_minimum_required(VERSION 3.13)
project(ManualBallLarus LANGUAGES C CXX)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# LLVM pass plugin (ONE target)
add_library(PathProfPass MODULE
  Plugin.cpp
  PathProfPass.cpp
  BuildBLTablePass.cpp
  DumpBLStaticPass.cpp
)
set_target_properties(PathProfPass PROPERTIES PREFIX "")

# Runtime library
add_library(bl_runtime STATIC bl_runtime.c)
