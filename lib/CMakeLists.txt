add_library(operators
            SHARED
            device_info.cpp
            act_quant.cpp
            zeros.cpp
            utils.cpp
            sum.cpp
            max.cpp
            mm.cpp
            rms_norm.cpp
            gemma_rms_norm.cpp
            fused_add_rms_norm.cpp
            rotary_embedding.cpp
            topk.cpp
            addmm.cpp
            nonzero.cpp
            contiguous.cpp
            cat.cpp
            bmm.cpp
            embedding.cpp
            argmax.cpp
            softmax.cpp
            sort.cpp
	          exponential_.cpp
            reshape_and_cache_flash.cpp
            flash_attn_varlen_func.cpp
            rwkv_mm_sparsity.cpp
            rwkv_ka_fusion.cpp
            copy.cpp
            accuracy_utils.cpp
            fp8_matmul.cpp
            fp8_matmul_direct.cpp
            index_fill.cpp)

if(FLAGGEMS_BACKEND STREQUAL "NPU" AND FLAGGEMS_ASCENDC_AVAILABLE)
  set(_index_fill_ascendc_build_dir
    "${CMAKE_CURRENT_BINARY_DIR}/index_fill_ascendc_external")
  set(_index_fill_ascendc_archive
    "${_index_fill_ascendc_build_dir}/lib/libindex_fill_ascendc_kernel.a")
  set(_index_fill_ascendc_include_dir
    "${_index_fill_ascendc_build_dir}/include/index_fill_ascendc_kernel")
  set(_index_fill_ascendc_launcher
    "${_index_fill_ascendc_include_dir}/aclrtlaunch_flag_gems_index_fill_fused_2d.h")
  file(MAKE_DIRECTORY "${_index_fill_ascendc_include_dir}")

  add_custom_command(
    OUTPUT "${_index_fill_ascendc_archive}"
    BYPRODUCTS "${_index_fill_ascendc_launcher}"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_index_fill_ascendc_build_dir}"
    COMMAND "${CMAKE_COMMAND}"
      -S "${CMAKE_CURRENT_SOURCE_DIR}/ascendc/index_fill"
      -B "${_index_fill_ascendc_build_dir}"
      -G "Unix Makefiles"
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME}"
      "-DASCENDC_CMAKE_DIR=${ASCENDC_CMAKE_DIR}"
      "-DSOC_VERSION=${SOC_VERSION}"
      "-DINDEX_FILL_KERNEL_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/index_fill_ascendc_kernel.cpp"
    COMMAND "${CMAKE_COMMAND}" --build "${_index_fill_ascendc_build_dir}"
      --target index_fill_ascendc_kernel --parallel 1
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/ascendc/index_fill/CMakeLists.txt"
      "${CMAKE_CURRENT_SOURCE_DIR}/index_fill_ascendc_kernel.cpp"
    COMMENT "Building the Ascend C index_fill kernel"
    VERBATIM)
  add_custom_target(index_fill_ascendc_external
    DEPENDS "${_index_fill_ascendc_archive}")

  add_library(index_fill_ascendc_kernel_bin STATIC IMPORTED GLOBAL)
  set_target_properties(index_fill_ascendc_kernel_bin PROPERTIES
    IMPORTED_LOCATION "${_index_fill_ascendc_archive}"
    INTERFACE_INCLUDE_DIRECTORIES "${_index_fill_ascendc_include_dir}")
  add_dependencies(index_fill_ascendc_kernel_bin index_fill_ascendc_external)

  target_sources(operators PRIVATE index_fill_ascendc.cpp)
  target_link_libraries(operators PRIVATE
    index_fill_ascendc_kernel_bin
    tiling_api
    nnopbase
    opapi
    register
    platform
    ascendalog
    dl)
  target_link_directories(operators PRIVATE "${ASCEND_HOME}/lib64")
  target_compile_definitions(operators PRIVATE FLAGGEMS_USE_ASCENDC)
  add_dependencies(operators index_fill_ascendc_external)
endif()

if (TRITON_GE_3P5)
  target_compile_definitions(operators PRIVATE TRITON_GE_3P5)
endif()

target_include_directories(operators
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# Backend conditional linking
if(FLAGGEMS_BACKEND STREQUAL "CUDA" OR FLAGGEMS_BACKEND STREQUAL "IX")
    target_link_libraries(operators
      PUBLIC Torch::Torch
      PRIVATE TritonJIT::triton_jit
      PRIVATE CUDA::cudart CUDA::cuda_driver)
elseif(FLAGGEMS_BACKEND STREQUAL "NPU")
    target_link_npu_libraries(operators)
    target_link_libraries(operators
      PUBLIC Torch::Torch
      PRIVATE TritonJIT::triton_jit)
elseif(FLAGGEMS_BACKEND STREQUAL "MUSA")
    target_link_musa_libraries(operators)
    target_link_libraries(operators
      PUBLIC Torch::Torch
      PRIVATE TritonJIT::triton_jit)
elseif(FLAGGEMS_BACKEND STREQUAL "GCU")
    target_link_gcu_libraries(operators)
    target_link_libraries(operators
      PUBLIC Torch::Torch
      PRIVATE TritonJIT::triton_jit)
elseif(FLAGGEMS_BACKEND STREQUAL HCU)
    target_link_hcu_libraries(operators)
    target_link_libraries(operators
      PUBLIC Torch::Torch
      PRIVATE TritonJIT::triton_jit)
    target_compile_options(operators PRIVATE -std=c++20)
endif()
target_compile_definitions(operators PRIVATE BACKEND_${FLAGGEMS_BACKEND})

add_library(FlagGems::operators ALIAS operators)


# ==============================================================================
# Pointwise dynamic C++ support (experimental feature, disabled by default)
# ==============================================================================
option(FLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP "Build pointwise_dynamic C++ support" OFF)
if(FLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP)
  add_subdirectory(pointwise_dynamic_cpp)
  target_sources(operators PRIVATE add.cpp div.cpp fill.cpp)
  target_link_libraries(operators PRIVATE pointwise_dynamic)
  target_compile_definitions(operators PRIVATE FLAGGEMS_POINTWISE_DYNAMIC)
endif()

if (FLAGGEMS_INSTALL)
  include(GNUInstallDirs)
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/flag_gems"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(TARGETS operators
    EXPORT FlagGemsTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(EXPORT FlagGemsTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagGems
    NAMESPACE FlagGems::
    FILE FlagGemsTargets.cmake)
  export(EXPORT FlagGemsTargets FILE "${PROJECT_BINARY_DIR}/FlagGemsTargets.cmake")
endif()
