/**
 * @file
 * @brief In-kernel timing ring: a per-CTA span recorder for Gantt profiling.
 *
 * Nsight sees one kernel launch as one bar and cannot separate concurrent warp
 * roles inside a single CTA. This records (timestamp, event_id, payload)
 * triples straight from the running kernel so the host can pair them into
 * labelled spans.
 *
 * Everything here is behind PROFILE_TIMINGS: `make PROFILE=1 ...` builds a
 * profiling .so, a plain `make` leaves zero profile instructions in the SASS.
 */
#pragma once

#ifdef PROFILE_TIMINGS

#include <cstdint>

#include "comm/device_clock.cuh"

#ifndef PROFILE_EVENTS_PER_BLOCK
#define PROFILE_EVENTS_PER_BLOCK 65536
#endif

// Per-reduction-step spans are ON by default. Tile-level spans alone are not
// useful here: one tile covers the whole K reduction (~400 us), and successive
// tiles run back to back, so every lane renders as one gapless band. The
// waiting -- who is starved on whom -- only becomes visible at the K step.
// Build with PROFILE_COARSE=1 to drop them and pay ~6 fewer emits per K step.
#if !defined(PROFILE_COARSE) && !defined(PROFILE_TIMINGS_FINE)
#define PROFILE_TIMINGS_FINE 1
#endif

namespace timings {

struct TimingRecord {
    uint64_t timestamp;  // %globaltimer, nanoseconds, device-wide monotonic
    uint32_t event_id;   // per-kernel enum
    uint32_t payload;    // pairing key: (role << 28) | sequence
};
static_assert(sizeof(TimingRecord) == 16, "TimingRecord must be one 128-bit store");

// Compile-time cap on emits per CTA. HBM cost is
// NUM_BLOCKS * EVENTS_PER_BLOCK * 16B (148 * 65536 * 16 == 155 MB here).
static constexpr int EVENTS_PER_BLOCK = PROFILE_EVENTS_PER_BLOCK;

// Roles sharing one CTA's head. Packed into the payload so that spans from
// warps running concurrently in the same CTA never pair against each other.
static constexpr uint32_t ROLE_CTA = 0;
static constexpr uint32_t ROLE_PRODUCER = 1;
static constexpr uint32_t ROLE_MMA = 2;
static constexpr uint32_t ROLE_EPILOGUE = 3;

__device__ __forceinline__ uint32_t pack_payload(uint32_t role, uint32_t seq) {
    return (role << 28) | (seq & 0x0FFFFFFFu);
}

__device__ __forceinline__ void emit_timing_impl(TimingRecord* buf,
                                                 uint32_t* s_head,
                                                 uint32_t event_id,
                                                 uint32_t payload) {
    if (buf == nullptr) return;
    uint32_t idx = atomicAdd(s_head, 1u);
    if (idx < EVENTS_PER_BLOCK) {
        // One STG.128. Writing the halves separately costs two STG.64 and can
        // tear the record for anyone reading mid-store.
        ulonglong2 rec;
        rec.x = comm::globaltimer();
        rec.y = ((uint64_t)event_id << 32) | (uint64_t)payload;
        reinterpret_cast<ulonglong2*>(buf)[(uint64_t)blockIdx.x * EVENTS_PER_BLOCK + idx] = rec;
    }
    // Overflow past the cap is a silent drop, never an out-of-bounds write.
}

}  // namespace timings

#define EMIT(buf, s_head, eid, role, seq)          \
    ::timings::emit_timing_impl((buf),             \
                                (s_head),          \
                                (uint32_t)(eid),   \
                                ::timings::pack_payload((role), (uint32_t)(seq)))

#endif  // PROFILE_TIMINGS
