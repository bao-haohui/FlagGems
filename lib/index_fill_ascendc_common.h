#pragma once

#include <stdint.h>

namespace flag_gems {
namespace index_fill_ascendc {

enum class IndexFillDtype : uint32_t {
  kFloat16 = 0,
  kBFloat16 = 1,
  kFloat32 = 2,
};

enum class IndexFillPath : uint32_t {
  kGeneral = 0,
  kDim0InplaceSmall = 1,
  kDim0InplaceSmallDeduplicate = 2,
  kDim0FunctionalSmallDirectMatch = 3,
  kDim0FunctionalSmallMembership = 4,
};

constexpr uint32_t kMinSupportedDim = 0;
constexpr uint32_t kMaxSupportedDim = 1;
constexpr uint32_t kMaxDimSize = 4096;
constexpr uint32_t kMaxTensorNumel = 0xFFFFFFFFU;
constexpr uint32_t kMinIndexCount = 8;
constexpr uint32_t kMaxIndexCount = 4096;
constexpr uint32_t kIndexAlignment = 8;
constexpr uint32_t kDim1VectorAlignment = 256;
constexpr uint32_t kColumnAlignment = 16;
constexpr uint32_t kMaxDim0InplaceSmallIndexCount = 256;
constexpr uint32_t kMaxDim0FunctionalSmallIndexCount = 256;
constexpr uint32_t kBlockCount = 40;

}  // namespace index_fill_ascendc
}  // namespace flag_gems
