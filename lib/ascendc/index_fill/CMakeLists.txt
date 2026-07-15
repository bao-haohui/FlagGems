cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(flaggems-index-fill-ascendc LANGUAGES CXX)

if(NOT ASCEND_CANN_PACKAGE_PATH OR NOT ASCENDC_CMAKE_DIR OR NOT SOC_VERSION)
  message(FATAL_ERROR "Ascend C build paths and SOC_VERSION are required")
endif()
if(NOT INDEX_FILL_KERNEL_SOURCE)
  message(FATAL_ERROR "INDEX_FILL_KERNEL_SOURCE is required")
endif()

include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
ascendc_library(index_fill_ascendc_kernel STATIC
  "${INDEX_FILL_KERNEL_SOURCE}")
ascendc_compile_definitions(index_fill_ascendc_kernel PRIVATE ASCENDC_DUMP=0)
