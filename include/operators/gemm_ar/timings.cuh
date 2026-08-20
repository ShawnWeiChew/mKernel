#pragma once

// In-kernel timing ring: a per-CTA event log that a Gantt renderer turns into
// a picture of who is waiting on whom inside a single kernel launch.
//
// Nsight sees one launch as one bar and cannot separate the concurrent warp
// roles inside it, which is exactly the question this kernel poses (is the
// producer starved, is the MMA warp blocked on tmem, is the epilogue late?).
// The trade is one shared-memory atomicAdd + one %globaltimer read + one
// 128-bit store per event, in exchange for exact intra-kernel spans.
//
// This header owns only the record format, the ring layout, and the emit
// path. Event identities belong to the kernel using it (see TimingEvent in
// src/gemm_ar_blackwell.cu).
//
// Build with -DPROFILE_TIMINGS (make gemm_ar_blackwell_profile) to compile the
// emit path in. Without it EMIT expands to nothing and no profiling
// instruction reaches the SASS.

#include <cstdint>

// Compile-time cap on events per CTA. Overflow past this is dropped by the
// bounds check in emit_timing_impl -- a truncated tail, never an out-of-bounds
// write into the neighbouring CTA's slot. HBM cost is
// NUM_BLOCKS * EVENTS_PER_BLOCK * 16B, i.e. ~155 MB at 148 blocks and 65536.
#ifndef PROFILE_EVENTS_PER_BLOCK
#define PROFILE_EVENTS_PER_BLOCK 65536
#endif

namespace mkernel_timings {

// Exactly 16 bytes so one emit is one STG.128. Do not add a fourth 32-bit
// field: it doubles the store count and the write cost for no information.
struct TimingRecord {
    uint64_t timestamp;  // %globaltimer, nanoseconds
    uint32_t event_id;   // kernel-defined enum
    uint32_t payload;    // pairing key (see pack_payload)
};
static_assert(sizeof(TimingRecord) == 16, "TimingRecord must stay one 128-bit store");

static constexpr int EVENTS_PER_BLOCK = PROFILE_EVENTS_PER_BLOCK;

#ifdef PROFILE_TIMINGS
static constexpr bool TIMINGS_COMPILED = true;
#else
static constexpr bool TIMINGS_COMPILED = false;
#endif

// Pairing key. All warps of a CTA share one head and therefore one slot range,
// so a bare sequence number would collide across warps and the pairer would
// match a begin from one warp to an end from another (huge or negative spans).
// Top 4 bits: warp id (16 warps max). Low 28 bits: per-warp sequence number.
// The host decodes the same split to put each warp on its own Gantt row.
__host__ __device__ __forceinline__ uint32_t pack_payload(int warp_id, uint32_t seq) {
    return ((uint32_t)warp_id << 28) | (seq & 0x0FFFFFFFu);
}

#ifdef PROFILE_TIMINGS

// Device-wide monotonic nanoseconds. %clock is per-SM and cannot be aligned
// between CTAs, so it is useless for a cross-CTA timeline.
__device__ __forceinline__ uint64_t dev_gtime() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// buf is the whole ring; this CTA owns [blockIdx.x * EVENTS_PER_BLOCK, +cap),
// so CTAs never contend with each other and the only atomic is the CTA-local
// head in shared memory (single-digit cycles under this contention).
__device__ __forceinline__ void emit_timing_impl(TimingRecord* buf,
                                                 uint32_t* s_head,
                                                 uint32_t event_id,
                                                 uint32_t payload) {
    uint32_t idx = atomicAdd(s_head, 1u);
    if (idx < (uint32_t)EVENTS_PER_BLOCK && buf != nullptr) {
        // Written as one ulonglong2. Storing the halves separately gets two
        // STG.64s -- twice the traffic, and a torn record if anything reads
        // the slot mid-store.
        ulonglong2 rec;
        rec.x = dev_gtime();
        rec.y = ((uint64_t)event_id << 32) | (uint64_t)payload;
        reinterpret_cast<ulonglong2*>(buf)[(uint64_t)blockIdx.x * EVENTS_PER_BLOCK + idx] = rec;
    }
}

#define EMIT(buf, s_head, eid, payload)                     \
    ::mkernel_timings::emit_timing_impl((buf),              \
                                        (s_head),           \
                                        (uint32_t)(eid),    \
                                        (uint32_t)(payload))
#else
#define EMIT(buf, s_head, eid, payload) ((void)0)
#endif  // PROFILE_TIMINGS

}  // namespace mkernel_timings
