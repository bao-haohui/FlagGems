# P0 is a standalone Ascend C probe.  It deliberately does not modify the
# FlagGems runtime, operator registry, or index_fill dispatch.
cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(index_fill_ascendc_p0 LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(INDEX_FILL_P0_NPU_ARCH "" CACHE STRING
    "Ascend C architecture for this artifact, for example dav-2201 or dav-3510")
set(INDEX_FILL_P0_NPU_ARCH_ID "" CACHE STRING
    "Numeric NpuArch reported by the matching runtime platform, for example 2201")

if(INDEX_FILL_P0_NPU_ARCH STREQUAL "" OR INDEX_FILL_P0_NPU_ARCH_ID STREQUAL "")
    message(FATAL_ERROR
        "Set both INDEX_FILL_P0_NPU_ARCH and INDEX_FILL_P0_NPU_ARCH_ID. "
        "Example: -DINDEX_FILL_P0_NPU_ARCH=dav-2201 -DINDEX_FILL_P0_NPU_ARCH_ID=2201")
endif()

set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}")
if(ASCEND_HOME STREQUAL "")
    message(FATAL_ERROR "ASCEND_HOME_PATH must point at the active CANN installation")
endif()

add_executable(index_fill_ascendc_p0
    op_host/index_fill_p0.asc
)

target_include_directories(index_fill_ascendc_p0 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${ASCEND_HOME}/aarch64-linux/include
    ${ASCEND_HOME}/aarch64-linux/asc/include
)

target_link_directories(index_fill_ascendc_p0 PRIVATE
    ${ASCEND_HOME}/lib64
    ${ASCEND_HOME}/aarch64-linux/lib64
)

target_link_libraries(index_fill_ascendc_p0 PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    graph_base
    ascendcl
    ascendc_runtime
    dl
    m
)

target_compile_definitions(index_fill_ascendc_p0 PRIVATE
    INDEX_FILL_P0_COMPILED_NPU_ARCH=${INDEX_FILL_P0_NPU_ARCH_ID}
)

target_compile_options(index_fill_ascendc_p0 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${INDEX_FILL_P0_NPU_ARCH}>
)

message(STATUS "index_fill P0 artifact: ${INDEX_FILL_P0_NPU_ARCH} (${INDEX_FILL_P0_NPU_ARCH_ID})")
