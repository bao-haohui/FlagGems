#pragma once

#include <cstdint>

// Shared host/device ABI for the P0 copy/fill probe.  P1 will replace this
// narrow payload with index_fill workload and path metadata.
enum class CopyFillP0Mode : uint32_t {
    kCopy = 0,
    kFill = 1,
};

struct CopyFillP0TilingData {
    uint32_t blockDim;
    uint32_t tileElements;
    uint32_t mode;
    uint32_t reserved;
    uint64_t totalElements;
    uint64_t elementsPerCore;
    uint64_t tailElements;
    float fillValue;
    uint32_t reserved1;
};
