#pragma once

// In-kernel timing profiler: a per-CTA ring of (timestamp, event, payload)
// records that renders as a Gantt chart of intra-kernel spans.
//
// Nsight sees one kernel launch as one bar and cannot separate concurrent CTA
// roles inside it, which is exactly what a warp-specialised persistent kernel
// needs. This trades ~50ns per emit for that resolution.
//
// Everything here is compiled away unless -DPROFILE_TIMINGS is set: the macros
// below expand to `((void)0)`, and because discarded macro arguments never
// reach the compiler, call sites may name profile-only enumerators and
// profile-only struct fields without their own #ifdef.
//
// Host side: python/timings.py unpacks the ring and renders it.

#include <cstdint>

namespace mkernel_timings {

// Compile-time cap on emits per CTA. Records past it are dropped by the bounds
// check in emit_timing_impl -- truncating a CTA's tail rather than corrupting
// its neighbour's slot. Override with -DMKERNEL_EVENTS_PER_BLOCK=N.
#ifndef MKERNEL_EVENTS_PER_BLOCK
#define MKERNEL_EVENTS_PER_BLOCK 65536
#endif
static constexpr int EVENTS_PER_BLOCK = MKERNEL_EVENTS_PER_BLOCK;

// The ring is partitioned per CTA, so warps sharing a CTA also share a slot and
// a head. What separates them again on the host -- and what keeps the pairing
// key unique when two warps are on the same sequence number -- is the warp id
// packed into the payload's top bits.
static constexpr int WARP_ID_SHIFT = 28;
static constexpr uint32_t SEQ_MASK = (1u << WARP_ID_SHIFT) - 1u;

struct TimingRecord {
    uint64_t timestamp;  // %globaltimer, nanoseconds
    uint32_t event_id;   // kernel-defined enum
    uint32_t payload;    // (warp_id << 28) | seq -- what begin/end pair on
};
static_assert(sizeof(TimingRecord) == 16,
              "A record must stay 16B so an emit is one STG.128; a fourth field "
              "doubles the write cost for nothing");

__device__ __forceinline__ uint32_t pack_payload(int warp_id, uint32_t seq) {
    return ((uint32_t)warp_id << WARP_ID_SHIFT) | (seq & SEQ_MASK);
}

// %globaltimer is device-wide monotonic ns -- the only clock that lets
// timestamps from CTAs on different SMs be compared. %clock is per-SM and
// cannot be aligned. It is *not* synchronised across GPUs: see the per-rank
// anchoring note in python/timings.py.
__device__ __forceinline__ uint64_t dev_gtime() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void emit_timing_impl(TimingRecord* buf,
                                                 uint32_t* s_head,
                                                 uint32_t event_id,
                                                 uint32_t payload) {
    // The head is per-CTA shared memory, so this atomic is single-digit cycles
    // and serialises cleanly across the CTA's warps. One head per warp would
    // hand every warp the same 0,1,2,... and alias them into the same slots.
    uint32_t idx = atomicAdd(s_head, 1u);
    if (buf != nullptr && idx < (uint32_t)EVENTS_PER_BLOCK) {
        // Written as one 128-bit store: two 64-bit stores would double the
        // traffic and can tear a record for a concurrent reader.
        ulonglong2 rec;
        rec.x = dev_gtime();
        rec.y = ((uint64_t)event_id << 32) | (uint64_t)payload;
        reinterpret_cast<ulonglong2*>(buf)[(uint64_t)blockIdx.x * EVENTS_PER_BLOCK + idx] = rec;
    }
}

}  // namespace mkernel_timings

#ifdef PROFILE_TIMINGS

// Declare the CTA's shared head and zero it. The caller must reach a
// __syncthreads() between this and its first emit; MKERNEL_TIMING_PROLOGUE
// does that for you.
#define MKERNEL_TIMING_HEAD_DECL()             \
    __shared__ uint32_t s_timing_head;         \
    if (threadIdx.x == 0) s_timing_head = 0u

#define MKERNEL_TIMING_PROLOGUE() \
    MKERNEL_TIMING_HEAD_DECL();   \
    __syncthreads()

// warp_id is the payload's discriminator, not necessarily warpid(): emit from
// one lane per logical role and pass that role's warp.
#define MKERNEL_EMIT(buf, warp_id, eid, seq)                       \
    ::mkernel_timings::emit_timing_impl(                           \
        (buf), &s_timing_head, (uint32_t)(eid),                    \
        ::mkernel_timings::pack_payload((warp_id), (uint32_t)(seq)))

// Emit only from the lanes matching `pred`. Use this rather than an `if` around
// MKERNEL_EMIT: the predicate is often profile-only state (or a real warp-vote
// instruction), and here it disappears with the emit instead of being left
// behind as a dangling condition.
#define MKERNEL_EMIT_IF(pred, buf, warp_id, eid, seq)          \
    do {                                                       \
        if (pred) MKERNEL_EMIT(buf, warp_id, eid, seq);        \
    } while (0)

#define MKERNEL_TIMING_ONLY(...) __VA_ARGS__

#else  // !PROFILE_TIMINGS

#define MKERNEL_TIMING_HEAD_DECL() ((void)0)
#define MKERNEL_TIMING_PROLOGUE() ((void)0)
#define MKERNEL_EMIT(buf, warp_id, eid, seq) ((void)0)
#define MKERNEL_EMIT_IF(pred, buf, warp_id, eid, seq) ((void)0)
#define MKERNEL_TIMING_ONLY(...)

#endif  // PROFILE_TIMINGS
