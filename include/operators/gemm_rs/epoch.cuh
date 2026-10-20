#pragma once

#include <cstdint>

namespace gemm_rs_blackwell {

// Modular comparison: valid when epochs/counters differ by less than 2^31.
// The exit barrier bounds rank skew; tile counts advance NUM_DEVICES at a time.
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr bool epoch_reached(uint32_t observed, uint32_t expected) {
    return uint32_t(observed - expected) < 0x80000000u;
}

}  // namespace gemm_rs_blackwell
