#include <algorithm>
#include <limits>
#include <string>

#include "flag_gems/backend_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {

using namespace triton_jit;

namespace {

int64_t product_after_dim(const at::Tensor& input, int64_t dim) {
  int64_t result = 1;
  for (int64_t i = dim + 1; i < input.dim(); ++i) {
    result *= input.size(i);
  }
  return result;
}

int64_t product_before_dim(const at::Tensor& input, int64_t dim) {
  int64_t result = 1;
  for (int64_t i = 0; i < dim; ++i) {
    result *= input.size(i);
  }
  return result;
}

int64_t normalize_dim(int64_t dim, int64_t ndim) {
  if (dim < 0) dim += ndim;
  return dim;
}

enum class IndexFillLauncherVariant {
  kAuto,
  kGeneral,
  kSmallInner,
};

IndexFillLauncherVariant index_fill_launcher_variant_from_name(
    const std::string& name) {
  if (name == "auto") return IndexFillLauncherVariant::kAuto;
  if (name == "general") return IndexFillLauncherVariant::kGeneral;
  if (name == "small_inner") return IndexFillLauncherVariant::kSmallInner;
  TORCH_CHECK(false, "unknown index_fill benchmark variant: ", name);
  return IndexFillLauncherVariant::kAuto;
}

}  // namespace

namespace {

at::Tensor& launch_index_fill_scalar_(at::Tensor& input,
                                      int64_t dim,
                                      const at::Tensor& index,
                                      const c10::Scalar& value,
                                      IndexFillLauncherVariant variant) {
  TORCH_CHECK_INDEX(input.dim() > 0,
                    "index_fill expects self to have at least one dimension");
  dim = normalize_dim(dim, input.dim());
  TORCH_CHECK_INDEX(dim >= 0 && dim < input.dim(),
                    "index_fill: dim out of range");
  TORCH_CHECK_INDEX(index.scalar_type() == at::kLong,
                    "index_fill_(): Expected dtype int64 for index.");
  TORCH_CHECK_INDEX(index.dim() <= 1,
                    "index_fill_(): Index is supposed to be a vector");
  TORCH_CHECK(index.device() == input.device(),
              "index and input must be on the same device");
  TORCH_CHECK(input.is_contiguous(),
              "C++ index_fill_ launcher only supports contiguous input");
  TORCH_CHECK(backend::isOnDevice(input),
              "input must be on the active backend device");

  const int64_t index_len = index.numel();
  if (input.numel() == 0 || index_len == 0) return input;
  const at::Tensor contiguous_index =
      index.is_contiguous() ? index : index.contiguous();

  const int64_t dim_size = input.size(dim);
  const int64_t inner_size = product_after_dim(input, dim);
  const int64_t outer_size = product_before_dim(input, dim);
  const int64_t outer_index_len = outer_size * index_len;
  const bool prefer_generic_small_inner =
      inner_size <= 4 && outer_size > 1 && dim_size > 8192 &&
      index_len > dim_size / 16;
  const bool small_inner_eligible =
      outer_size > 1 && inner_size > 1 && inner_size <= 4;
  TORCH_CHECK(
      variant == IndexFillLauncherVariant::kAuto || small_inner_eligible,
      "index_fill benchmark variant requires outer_size > 1 and inner_size in [2, 4]");

  c10::DeviceGuard guard(input.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  if (variant == IndexFillLauncherVariant::kAuto && outer_size == 1 &&
      inner_size == 1) {
    constexpr int64_t BLOCK = 512;
    const unsigned int grid_x =
        static_cast<unsigned int>((index_len + BLOCK - 1) / BLOCK);

    static const TritonJITFunction& one_dim_kernel =
        TritonJITFunction::get_instance(
            (utils::get_triton_src_path() / "index_fill.py").string(),
            "index_fill_contiguous_scalar_1d_kernel");

    one_dim_kernel(raw_stream,
                   grid_x,
                   1,
                   1,
                   4,
                   0,
                   input,
                   contiguous_index,
                   value,
                   index_len,
                   dim_size,
                   BLOCK);
    return input;
  }

  if (variant == IndexFillLauncherVariant::kAuto && outer_size > 1 &&
      inner_size == 1) {
    const int64_t block_m = index_len <= 16 ? 512 : 1024;
    const unsigned int grid_x =
        static_cast<unsigned int>(
            (outer_index_len + block_m - 1) / block_m);

    static const TritonJITFunction& inner1_kernel =
        TritonJITFunction::get_instance(
            (utils::get_triton_src_path() / "index_fill.py").string(),
            "index_fill_contiguous_scalar_inner1_kernel");

    inner1_kernel(raw_stream,
                  grid_x,
                  1,
                  1,
                  4,
                  0,
                  input,
                  contiguous_index,
                  value,
                  outer_index_len,
                  index_len,
                  dim_size,
                  block_m);
    return input;
  }

  const bool inner3_element_dtype =
      input.scalar_type() == at::kHalf ||
      input.scalar_type() == at::kBFloat16 ||
      input.scalar_type() == at::kFloat;
  const int64_t int32_max = std::numeric_limits<int32_t>::max();
  if (variant == IndexFillLauncherVariant::kAuto && outer_size > 1 &&
      inner_size == 3 && !prefer_generic_small_inner &&
      inner3_element_dtype &&
      input.numel() <= int32_max && index_len <= int32_max &&
      index_len <= int32_max / 3 && dim_size <= int32_max &&
      outer_size <= int32_max) {
    constexpr int32_t BLOCK = 128;
    const int32_t row_elements = static_cast<int32_t>(index_len * 3);
    const unsigned int grid_x =
        static_cast<unsigned int>((row_elements + BLOCK - 1) / BLOCK);
    const unsigned int grid_y = static_cast<unsigned int>(outer_size);

    static const TritonJITFunction& inner3_element_kernel =
        TritonJITFunction::get_instance(
            (utils::get_triton_src_path() / "index_fill.py").string(),
            "index_fill_contiguous_scalar_inner3_element_kernel");

    inner3_element_kernel(raw_stream,
                          grid_x,
                          grid_y,
                          1,
                          4,
                          0,
                          input,
                          contiguous_index,
                          value,
                          row_elements,
                          static_cast<int32_t>(dim_size),
                          BLOCK);
    return input;
  }

  if (variant == IndexFillLauncherVariant::kSmallInner ||
      (variant == IndexFillLauncherVariant::kAuto && small_inner_eligible &&
       !prefer_generic_small_inner)) {
    constexpr int64_t BLOCK_M = 256;
    const unsigned int grid_x =
        static_cast<unsigned int>((outer_index_len + BLOCK_M - 1) / BLOCK_M);

    static const TritonJITFunction& small_inner_kernel =
        TritonJITFunction::get_instance(
            (utils::get_triton_src_path() / "index_fill.py").string(),
            "index_fill_contiguous_scalar_small_inner_flat_kernel");

    small_inner_kernel(raw_stream,
                       grid_x,
                       1,
                       1,
                       8,
                       0,
                       input,
                       contiguous_index,
                       value,
                       outer_index_len,
                       index_len,
                       dim_size,
                       BLOCK_M,
                       inner_size);
    return input;
  }

  constexpr int64_t BLOCK_SIZE = 512;
  int64_t block_n = 1;
  int64_t block_m = BLOCK_SIZE;
  if (inner_size >= 128) {
    block_n = std::min<int64_t>(128, utils::next_power_of_2(inner_size));
    block_m = 16;
  } else if (inner_size > 1) {
    block_n = std::min<int64_t>(64, utils::next_power_of_2(inner_size));
    if (inner_size > 4) {
      block_m = std::max<int64_t>(1, BLOCK_SIZE / block_n);
    }
  }

  const unsigned int grid_x =
      static_cast<unsigned int>((outer_index_len + block_m - 1) / block_m);
  const unsigned int grid_y =
      static_cast<unsigned int>((inner_size + block_n - 1) / block_n);

  static const TritonJITFunction& kernel = TritonJITFunction::get_instance(
      (utils::get_triton_src_path() / "index_fill.py").string(),
      "index_fill_contiguous_scalar_kernel");

  const bool use_int32_indexing =
      input.numel() <= int32_max && outer_index_len <= int32_max &&
      index_len <= int32_max && dim_size <= int32_max &&
      inner_size <= int32_max;
  if (use_int32_indexing) {
    kernel(raw_stream,
           grid_x,
           grid_y,
           1,
           4,
           0,
           input,
           contiguous_index,
           value,
           static_cast<int32_t>(outer_index_len),
           static_cast<int32_t>(index_len),
           static_cast<int32_t>(dim_size),
           static_cast<int32_t>(inner_size),
           static_cast<int32_t>(block_m),
           static_cast<int32_t>(block_n));
  } else {
    kernel(raw_stream,
           grid_x,
           grid_y,
           1,
           4,
           0,
           input,
           contiguous_index,
           value,
           outer_index_len,
           index_len,
           dim_size,
           inner_size,
           block_m,
           block_n);
  }

  return input;
}

}  // namespace

at::Tensor& index_fill_scalar_(at::Tensor& input,
                              int64_t dim,
                              const at::Tensor& index,
                              const c10::Scalar& value) {
  return launch_index_fill_scalar_(
      input, dim, index, value, IndexFillLauncherVariant::kAuto);
}

#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX)
at::Tensor index_fill_scalar_benchmark(const at::Tensor& input,
                                       int64_t dim,
                                       const at::Tensor& index,
                                       const c10::Scalar& value,
                                       const std::string& variant) {
  at::Tensor output = input.clone();
  launch_index_fill_scalar_(
      output, dim, index, value, index_fill_launcher_variant_from_name(variant));
  return output;
}

at::Tensor& index_fill_scalar_benchmark_(at::Tensor& input,
                                         int64_t dim,
                                         const at::Tensor& index,
                                         const c10::Scalar& value,
                                         const std::string& variant) {
  return launch_index_fill_scalar_(
      input, dim, index, value, index_fill_launcher_variant_from_name(variant));
}
#endif

at::Tensor index_fill_scalar(const at::Tensor& input,
                             int64_t dim,
                             const at::Tensor& index,
                             const c10::Scalar& value) {
#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX)
  if (input.is_contiguous()) {
    c10::DeviceGuard guard(input.device());
    at::Tensor output = at::empty_like(input);
    if (input.nbytes() > 0) {
      backend::StreamType stream = backend::getCurrentStream();
      backend::RawStreamType raw_stream = backend::getRawStream(stream);
      CUresult status = cuMemcpyDtoDAsync(
          reinterpret_cast<CUdeviceptr>(output.data_ptr()),
          reinterpret_cast<CUdeviceptr>(input.data_ptr()),
          input.nbytes(),
          raw_stream);
      TORCH_CHECK(status == CUDA_SUCCESS,
                  "index_fill device-to-device copy failed with CUDA error ",
                  static_cast<int>(status));
    }
    index_fill_scalar_(output, dim, index, value);
    return output;
  }
#endif

  at::Tensor output = input.clone();
  index_fill_scalar_(output, dim, index, value);
  return output;
}

}  // namespace flag_gems
