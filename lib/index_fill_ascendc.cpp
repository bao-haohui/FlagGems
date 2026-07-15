#include <ATen/ATen.h>
#include <acl/acl.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/BFloat16.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <limits>

#include "aclrtlaunch_flag_gems_index_fill_fused_2d.h"
#include "flag_gems/operators.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace flag_gems {
namespace {

  constexpr int64_t kMaxDimSize = 4096;
  constexpr int64_t kMinVectorElements = 256;
  constexpr int64_t kMaxIndexCount = 4096;
  constexpr int64_t kIndexAlignment = 8;
  constexpr int64_t kColumnAlignment = 16;
  constexpr int64_t kMaxDim0InplaceSmallIndexCount = 256;
  constexpr int64_t kMaxDim0FunctionalSmallIndexCount = 256;
  constexpr uint32_t kBlockDim = 40;
  constexpr uint32_t kGeneralPath = 0;
  constexpr uint32_t kDim0InplaceSmallPath = 1;
  constexpr uint32_t kDim0InplaceSmallDeduplicatePath = 2;
  constexpr uint32_t kDim0FunctionalSmallDirectPath = 3;
  constexpr uint32_t kDim0FunctionalSmallMembershipPath = 4;

  bool check_index_bounds(const at::Tensor &index, int64_t dim_size, aclrtStream stream, bool check_unique) {
    std::array<int64_t, kMaxIndexCount> host_index;
    std::bitset<kMaxDimSize> seen;
    bool all_unique = true;
    const size_t index_bytes = index.numel() * sizeof(int64_t);
    auto status = aclrtMemcpyAsync(host_index.data(),
                                   index_bytes,
                                   index.data_ptr(),
                                   index_bytes,
                                   ACL_MEMCPY_DEVICE_TO_HOST,
                                   stream);
    TORCH_CHECK(status == ACL_SUCCESS, "Ascend C index_fill index copy failed with status ", status);
    status = aclrtSynchronizeStream(stream);
    TORCH_CHECK(status == ACL_SUCCESS,
                "Ascend C index_fill index synchronization failed with status ",
                status);
    for (int64_t i = 0; i < index.numel(); ++i) {
      const int64_t index_value = host_index[i];
      TORCH_CHECK_INDEX(index_value >= -dim_size && index_value < dim_size, "index out of range in self");
      if (check_unique) {
        const int64_t normalized_index = index_value < 0 ? index_value + dim_size : index_value;
        all_unique = all_unique && !seen.test(normalized_index);
        seen.set(normalized_index);
      }
    }
    return all_unique;
  }

  void check_fast_path_args(const at::Tensor &input, int64_t dim, const at::Tensor &index) {
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
                    input.scalar_type() == at::kFloat,
                "Ascend C index_fill fast path requires float16, bfloat16, or float32 input");
    TORCH_CHECK(input.dim() == 2 && input.size(0) > 0 && input.size(1) > 0 &&
                    input.numel() <= std::numeric_limits<uint32_t>::max(),
                "Ascend C index_fill fast path received an unsupported input shape");
    TORCH_CHECK((dim == 0 || dim == 1) && input.size(dim) <= kMaxDimSize,
                "Ascend C index_fill fast path received an unsupported dimension");
    TORCH_CHECK(index.scalar_type() == at::kLong && index.numel() >= kIndexAlignment &&
                    index.numel() <= kMaxIndexCount && index.numel() % kIndexAlignment == 0,
                "Ascend C index_fill fast path received an unsupported int64 index length");
    TORCH_CHECK(input.is_contiguous() && index.is_contiguous(),
                "Ascend C index_fill fast path requires contiguous tensors");
    TORCH_CHECK(input.device() == index.device(), "input and index must be on the same device");
  }

  void launch_index_fill(const at::Tensor &input,
                         int64_t dim,
                         const at::Tensor &index,
                         at::Tensor &output,
                         const c10::Scalar &value,
                         bool inplace) {
    const int64_t dim_size = input.size(dim);
    c10::DeviceGuard guard(input.device());
    auto stream = c10_npu::getCurrentNPUStream(input.get_device()).stream(true);
    const bool dim0_inplace_small_candidate =
        dim == 0 && inplace && index.numel() <= kMaxDim0InplaceSmallIndexCount;
    const bool use_dim0_functional_small =
        dim == 0 && !inplace && index.numel() <= kMaxDim0FunctionalSmallIndexCount;
    const bool all_unique = check_index_bounds(index, dim_size, stream, dim0_inplace_small_candidate);
    const bool use_dim0_inplace_small =
        dim0_inplace_small_candidate && (all_unique || index.numel() == kIndexAlignment);
    at::Tensor membership;
    void *membership_ptr = nullptr;
    if (!use_dim0_inplace_small && !use_dim0_functional_small) {
      int64_t membership_elements = (dim_size + kColumnAlignment - 1) / kColumnAlignment * kColumnAlignment;
      if (dim == 1 && membership_elements < kMinVectorElements) {
        membership_elements = kMinVectorElements;
      }
      membership = at::empty({membership_elements}, input.options().dtype(at::kHalf));
      membership_ptr = membership.data_ptr();
    }
    const float value_fp32 = static_cast<float>(value.toDouble());
    const uint32_t value_bf16_bits = c10::BFloat16(value_fp32).x;
    const uint32_t rows = input.size(0);
    const uint32_t cols = input.size(1);
    const uint32_t index_count = index.numel();
    const uint32_t kernel_dim = dim;
    const uint32_t kernel_inplace = inplace ? 1 : 0;
    uint32_t path_code = kGeneralPath;
    uint32_t block_dim = kBlockDim;
    if (use_dim0_inplace_small) {
      path_code = all_unique ? kDim0InplaceSmallPath : kDim0InplaceSmallDeduplicatePath;
      block_dim = all_unique ? std::min<uint32_t>(index_count / kIndexAlignment, kBlockDim) : 1;
    } else if (use_dim0_functional_small) {
      path_code = index_count <= 16 ? kDim0FunctionalSmallDirectPath : kDim0FunctionalSmallMembershipPath;
      block_dim = std::min<uint32_t>(rows, kBlockDim);
    }

    uint32_t dtype_code = 0;
    switch (input.scalar_type()) {
      case at::kHalf:
        dtype_code = 0;
        break;
      case at::kBFloat16:
        dtype_code = 1;
        break;
      case at::kFloat:
        dtype_code = 2;
        break;
      default:
        TORCH_CHECK(false, "unsupported Ascend C index_fill dtype");
    }

    auto launch = [stream,
                   input_ptr = const_cast<void *>(input.data_ptr()),
                   index_ptr = const_cast<void *>(index.data_ptr()),
                   output_ptr = output.data_ptr(),
                   membership_ptr,
                   value_fp32,
                   value_bf16_bits,
                   dtype_code,
                   rows,
                   cols,
                   index_count,
                   kernel_dim,
                   kernel_inplace,
                   path_code,
                   block_dim]() -> int {
      ACLRT_LAUNCH_KERNEL(flag_gems_index_fill_fused_2d)(block_dim,
                                                         stream,
                                                         input_ptr,
                                                         index_ptr,
                                                         output_ptr,
                                                         membership_ptr,
                                                         value_fp32,
                                                         value_bf16_bits,
                                                         dtype_code,
                                                         rows,
                                                         cols,
                                                         index_count,
                                                         kernel_dim,
                                                         kernel_inplace,
                                                         path_code,
                                                         block_dim);
      return 0;
    };
    at_npu::native::OpCommand::RunOpApi("flag_gems_index_fill_fused_2d", launch);
  }

}  // namespace

at::Tensor index_fill_ascendc_scalar(const at::Tensor &input,
                                     int64_t dim,
                                     const at::Tensor &index,
                                     const c10::Scalar &value) {
  check_fast_path_args(input, dim, index);
  at::Tensor output = at::empty_like(input);
  launch_index_fill(input, dim, index, output, value, false);
  return output;
}

at::Tensor &index_fill_ascendc_scalar_(at::Tensor &input,
                                       int64_t dim,
                                       const at::Tensor &index,
                                       const c10::Scalar &value) {
  check_fast_path_args(input, dim, index);
  launch_index_fill(input, dim, index, input, value, true);
  return input;
}

}  // namespace flag_gems
