#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <limits>

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <acl/acl.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
namespace {

int32_t choose_block_n(int64_t columns) {
  if (columns <= 256) return 256;
  if (columns <= 512) return 512;
  if (columns <= 1024) return 1024;
  if (columns <= 2048) return 2048;
  return 4096;
}

std::mutex kernel_path_mutex;
std::string configured_kernel_path;

void configure_kernel_path(const std::string& path) {
  TORCH_CHECK(std::filesystem::is_regular_file(path), "index_fill Triton source not found: ", path);
  std::lock_guard<std::mutex> lock(kernel_path_mutex);
  TORCH_CHECK(configured_kernel_path.empty() || configured_kernel_path == path,
              "index_fill Triton source path was already configured differently");
  configured_kernel_path = path;
}

std::string get_configured_kernel_path() {
  std::lock_guard<std::mutex> lock(kernel_path_mutex);
  TORCH_CHECK(!configured_kernel_path.empty(),
              "index_fill Triton source path has not been configured");
  return configured_kernel_path;
}

struct PreparedIndex {
  at::Tensor device_index;
  bool has_negative;
};

void validate_input(const at::Tensor& input, const at::Tensor& index) {
  TORCH_CHECK_INDEX(input.dim() == 2,
                    "_index_fill_dim0_npu only supports two-dimensional input");
  TORCH_CHECK(input.is_contiguous(),
              "_index_fill_dim0_npu only supports contiguous input");
  TORCH_CHECK(input.is_privateuseone(),
              "_index_fill_dim0_npu expects an NPU input");
  TORCH_CHECK_INDEX(index.scalar_type() == at::kLong,
                    "index_fill_(): Expected dtype int64 for index.");
  TORCH_CHECK_INDEX(index.dim() <= 1,
                    "index_fill_(): Index is supposed to be a vector");
  TORCH_CHECK(index.device() == input.device(),
              "index and input must be on the same device");
  TORCH_CHECK(input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16 ||
                  input.scalar_type() == at::kFloat,
              "_index_fill_dim0_npu supports float16, bfloat16, and float32 input");
}

PreparedIndex prepare_index(const at::Tensor& input, const at::Tensor& index) {
  at::Tensor device_index = index.contiguous();
  if (device_index.numel() == 0) return {std::move(device_index), false};

  // This synchronous host check preserves strict index_fill error semantics.
  at::Tensor host_index = device_index.cpu();
  const auto* values = host_index.data_ptr<int64_t>();
  const int64_t row_count = input.size(0);
  bool has_negative = false;
  for (int64_t i = 0; i < host_index.numel(); ++i) {
    const int64_t value = values[i];
    TORCH_CHECK_INDEX(value >= -row_count && value < row_count,
                      "index ", value,
                      " is out of bounds for dimension 0 with size ", row_count);
    has_negative |= value < 0;
  }
  return {std::move(device_index), has_negative};
}

void launch_row_fill(at::Tensor& output, const at::Tensor& index,
                     const c10::Scalar& value, bool has_negative) {
  const int64_t row_count = output.size(0);
  const int64_t row_width = output.size(1);
  const int64_t index_len = index.numel();
  if (output.numel() == 0 || index_len == 0) return;

  const int32_t block_n = choose_block_n(row_width);
  const uint32_t grid_x = static_cast<uint32_t>(index_len);
  const uint32_t grid_y = static_cast<uint32_t>((row_width + block_n - 1) / block_n);
  const bool use_int32 = output.numel() <= std::numeric_limits<int32_t>::max();

  c10::DeviceGuard guard(output.device());
  const auto stream = c10_npu::getCurrentNPUStream(output.device().index());
  const aclrtStream raw_stream = stream.stream();
  static const std::string kernel_path = get_configured_kernel_path();
  static const triton_jit::TritonJITFunction& kernel =
      triton_jit::TritonJITFunction::get_instance(
          kernel_path.c_str(), "index_fill_dim0_row_kernel");

  if (use_int32) {
    kernel(raw_stream, grid_x, grid_y, 1, 4, 0, output, index, value.toFloat(),
           static_cast<int32_t>(row_count), static_cast<int32_t>(row_width),
           static_cast<int32_t>(index_len), static_cast<int32_t>(has_negative),
           static_cast<int32_t>(true), block_n);
    return;
  }
  kernel(raw_stream, grid_x, grid_y, 1, 4, 0, output, index, value.toFloat(),
         row_count, row_width, index_len, static_cast<int32_t>(has_negative),
         static_cast<int32_t>(false), block_n);
}

at::Tensor& index_fill_dim0_npu_(at::Tensor& input, const at::Tensor& index,
                                 const c10::Scalar& value) {
  validate_input(input, index);
  PreparedIndex prepared = prepare_index(input, index);
  launch_row_fill(input, prepared.device_index, value, prepared.has_negative);
  return input;
}

at::Tensor index_fill_dim0_npu(const at::Tensor& input, const at::Tensor& index,
                                const c10::Scalar& value) {
  validate_input(input, index);
  PreparedIndex prepared = prepare_index(input, index);
  at::Tensor output = at::empty_like(input);
  c10::DeviceGuard guard(input.device());
  const auto stream = c10_npu::getCurrentNPUStream(input.device().index());
  TORCH_CHECK(
      aclrtMemcpyAsync(
          output.data_ptr(), output.numel() * output.element_size(), input.data_ptr(),
          input.numel() * input.element_size(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream.stream()) ==
          ACL_SUCCESS,
      "aclrtMemcpyAsync D2D copy failed");
  launch_row_fill(output, prepared.device_index, value, prepared.has_negative);
  return output;
}

}  // namespace
}  // namespace flag_gems

TORCH_LIBRARY_FRAGMENT(flag_gems, m) {
  m.def("_index_fill_dim0_npu(Tensor input, Tensor index, Scalar value) -> Tensor");
  m.def("_index_fill_dim0_npu_(Tensor(a!) input, Tensor index, Scalar value) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(flag_gems, PrivateUse1, m) {
  m.impl("_index_fill_dim0_npu", TORCH_FN(flag_gems::index_fill_dim0_npu));
  m.impl("_index_fill_dim0_npu_", TORCH_FN(flag_gems::index_fill_dim0_npu_));
}

PYBIND11_MODULE(_index_fill_npu_launcher, m) {
  m.def("configure_kernel_path", &flag_gems::configure_kernel_path);
}
