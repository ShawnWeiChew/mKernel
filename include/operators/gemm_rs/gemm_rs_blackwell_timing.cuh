#pragma once

#ifdef PROFILE_TIMINGS

#include <cstdint>

#include "common/timings.cuh"

namespace gemm_rs_blackwell {

using TimingRecord = timing_profile::TimingRecord;
static constexpr int EVENTS_PER_BLOCK = timing_profile::EVENTS_PER_BLOCK;

// These values are persisted in profile .npz files: append only. Events are
// chained milestones, so the end of one phase is the begin of the next and all
// milestones of one tile carry the same payload. The phase tables live in
// python/timings.py (LAYOUTS["gemm_rs_blackwell"]).
enum TimingEvent : uint32_t {
    // Every role leader: kernel entry -> pipeline ready.
    EV_SETUP_BEGIN = 0,
    EV_SETUP_DONE = 1,
    // Producer (warp 4): payload seq = cluster tile index.
    EV_LOAD_TILE_BEGIN = 2,
    EV_LOAD_TILE_ISSUED = 3,
    // MMA (warps 5-6, cta_rank 0 only): payload seq = cluster tile index.
    EV_MMA_TILE_BEGIN = 4,
    EV_MMA_TMEM_FREE = 5,
    EV_MMA_COMMITTED = 6,
    // Epilogue (thread 0): payload seq = tile index * CONSUMER_WARPS + consumer.
    EV_EPI_SUBTILE_BEGIN = 7,
    EV_EPI_MMA_DONE = 8,
    EV_EPI_TMEM_LOADED = 9,
    EV_EPI_STORES_ISSUED = 10,
    EV_EPI_STORE_DRAINED = 11,
    EV_EPI_SIGNALED = 12,
    EV_EPI_TEARDOWN_BEGIN = 13,
    EV_EPI_TEARDOWN_DONE = 14,
    // Reduce-scatter (warp 7, lane 0): payload seq = per-device tile index.
    EV_RS_TILE_BEGIN = 15,
    EV_RS_READY = 16,
    EV_RS_TILE_DONE = 17,
};

struct TimingEventName {
    const char* name;
    TimingEvent id;
};

// Exported to Python as TIMING_EVENTS; names drop the EV_ prefix.
inline constexpr TimingEventName TIMING_EVENT_NAMES[] = {
    {"SETUP_BEGIN", EV_SETUP_BEGIN},
    {"SETUP_DONE", EV_SETUP_DONE},
    {"LOAD_TILE_BEGIN", EV_LOAD_TILE_BEGIN},
    {"LOAD_TILE_ISSUED", EV_LOAD_TILE_ISSUED},
    {"MMA_TILE_BEGIN", EV_MMA_TILE_BEGIN},
    {"MMA_TMEM_FREE", EV_MMA_TMEM_FREE},
    {"MMA_COMMITTED", EV_MMA_COMMITTED},
    {"EPI_SUBTILE_BEGIN", EV_EPI_SUBTILE_BEGIN},
    {"EPI_MMA_DONE", EV_EPI_MMA_DONE},
    {"EPI_TMEM_LOADED", EV_EPI_TMEM_LOADED},
    {"EPI_STORES_ISSUED", EV_EPI_STORES_ISSUED},
    {"EPI_STORE_DRAINED", EV_EPI_STORE_DRAINED},
    {"EPI_SIGNALED", EV_EPI_SIGNALED},
    {"EPI_TEARDOWN_BEGIN", EV_EPI_TEARDOWN_BEGIN},
    {"EPI_TEARDOWN_DONE", EV_EPI_TEARDOWN_DONE},
    {"RS_TILE_BEGIN", EV_RS_TILE_BEGIN},
    {"RS_READY", EV_RS_READY},
    {"RS_TILE_DONE", EV_RS_TILE_DONE},
};

// The ring head is initialized alongside the mbarriers, after kernel entry, so
// the entry timestamp is captured in a register and emitted retroactively.
__device__ __forceinline__ void emit_setup(TimingRecord* records,
                                           uint32_t* head,
                                           uint64_t entry_ns,
                                           int warp_id) {
    const uint32_t payload = timing_profile::pack_payload(warp_id, 0);
    timing_profile::emit_at(records, head, entry_ns, EV_SETUP_BEGIN, payload);
    timing_profile::emit(records, head, EV_SETUP_DONE, payload);
}

}  // namespace gemm_rs_blackwell

#endif  // PROFILE_TIMINGS
