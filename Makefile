# Makefile — single-config build for 5 multinode kernels
#
# Usage:
#   make all       — build all 5 .so's into build/
#   make ENABLE_DISPATCH_GEMM_BLACKWELL=1 all
#                  — also build the intra-node Blackwell dispatch+GEMM kernel
#   make dispatch-gemm-blackwell SPECIALIZATION=sm|warp
#                  — build the selected 8-GPU B300 implementation
#   make run-dispatch-gemm-blackwell SPECIALIZATION=sm|warp
#                  — build and benchmark the selected implementation
#   make check     — run correctness check across all 5 kernels
#   make bench     — run wall-time bench across all 5 kernels
#   make plots     — regenerate TFLOPS bar charts under plots/
#   make clean     — remove build/

# === Backend selection ===
#
# Two backends are supported:
#   BACKEND=efa  → AWS EFA SRD via libibverbs+efadv (default)
#   BACKEND=cx7  → ConnectX-7 RC via libibverbs (InfiniBand / RoCE)
#
# Override with: `make BACKEND=cx7 all`
BACKEND ?= efa
ifeq ($(BACKEND),efa)
    BACKEND_DEFINES := -DINTERNODE_BACKEND_EFA
    BACKEND_LIBS    := -L$(EFA_HOME)/lib -lfabric -libverbs -lefa
else ifeq ($(BACKEND),cx7)
    BACKEND_DEFINES := -DINTERNODE_BACKEND_IBVERBS
    BACKEND_LIBS    := -libverbs
else
    $(error Unknown BACKEND=$(BACKEND). Use BACKEND=efa or BACKEND=cx7.)
endif

undefine BACKEND
undefine BACKEND_DEFINES
undefine BACKEND_LIBS

# === Target GPU ===
#   GPU=hopper    → sm_90a, wgmma MMA path (default, upstream behaviour)
#   GPU=blackwell → sm_103a, tcgen05 MMA path (B300; gemm_rs only so far)
GPU ?= hopper
ifeq ($(GPU),blackwell)
    ARCH              := -gencode arch=compute_103a,code=sm_103a
    ARCH_DEFINES      := -DKITTENS_SM10X -DKITTENS_BLACKWELL -DMKERNEL_TCGEN05
    DEFAULT_CUDA_HOME := /usr/local/cuda-13.1
    # conda forces a host compiler through NVCC_PREPEND_FLAGS/CXX on some boxes,
    # which makes nvcc miss system headers; pin the system g++.
    CCBIN             := -ccbin /usr/bin/g++
else ifeq ($(GPU),hopper)
    ARCH              := -gencode arch=compute_90a,code=sm_90a
    ARCH_DEFINES      := -DKITTENS_HOPPER
    DEFAULT_CUDA_HOME := /usr/local/cuda-12.9
    CCBIN             :=
else
    $(error Unknown GPU=$(GPU). Use GPU=hopper or GPU=blackwell.)
endif

# === Tooling ===
CUDA_HOME       ?= $(DEFAULT_CUDA_HOME)
EFA_HOME        ?= /opt/amazon/efa
NVCC            := $(CUDA_HOME)/bin/nvcc
# Python with torch installed. Override with `PYTHON=/path/to/python`.
PYTHON          ?= python3

# === Include paths (must precede LDFLAGS — TORCH_LIB feeds both) ===
HERE            := $(abspath .)
INC_RELEASE     := -I$(HERE)/include
ifeq ($(BACKEND),efa)
    INC_EFA     := -I$(EFA_HOME)/include
else
    INC_EFA     :=
endif
PY_INC          := $(shell $(PYTHON) -c "import sysconfig; print('-I'+sysconfig.get_path('include'))")
TORCH_INC       := $(shell $(PYTHON) -c "import torch.utils.cpp_extension as e; print(' '.join('-I'+p for p in e.include_paths()))")
TORCH_LIB       := $(shell $(PYTHON) -c "import torch.utils.cpp_extension as e; print(e.library_paths()[0])")

# INTRA_NUM_DEVICES = GPUs per logical node (multicast group size). Default 8
# matches an 8-GPU-per-node deployment. Override to test emulated multinode
# (e.g. `make INTRA_NUM_DEVICES=4 all` for 4 GPUs / "node").
INTRA_NUM_DEVICES ?= 8
COMMON_DEFINES  := $(ARCH_DEFINES) -DINTRA_NUM_DEVICES=$(INTRA_NUM_DEVICES) $(BACKEND_DEFINES)
COMMON_FLAGS    := -O3 -std=c++20 --use_fast_math --extended-lambda --expt-relaxed-constexpr $(ARCH) $(CCBIN)
LDFLAGS         := -shared -lcuda $(BACKEND_LIBS) \
                   -L$(TORCH_LIB) -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -ltorch_python \
                   -Xlinker -rpath -Xlinker $(TORCH_LIB) -L$(CUDA_HOME)/lib

COMMON_INC      := $(INC_RELEASE) $(INC_EFA) $(TORCH_INC) $(PY_INC)

# === Per-kernel constants (passed via -D, no env-var lookups) ===
#
# Note: keep per-kernel compile-time constants here until the corresponding
# source paths no longer need build-time specialization.
#
# Failed-experiment flags are NOT defined here (HYBRID, MERGED_COMM,
# PUSH_NVL_FANOUT, DISPATCH_DONATE_INTER_SEND, ACTIVITY_TRACE, etc.) so
# their #ifdef branches stay disabled.
DEFS_ag_gemm        :=
# Arrival-flag layout is now a runtime flag (SessionConfig.use_arrival_queue);
# gemm_ar's session shim sets it to true. No compile-time switch needed.
DEFS_gemm_ar        :=

TK_MOE_NUM_NODES ?= 2
DEFS_dispatch_gemm  := -DTK_MOE_H=7168 -DTK_MOE_I=2048 -DTK_MOE_TOP_K=8 -DTK_MOE_NUM_EXPERTS=256 -DTK_MOE_NUM_NODES=$(TK_MOE_NUM_NODES)
DEFS_dispatch_gemm_blackwell := -DTK_MOE_H=7168 -DTK_MOE_I=2048 -DTK_MOE_TOP_K=8 -DTK_MOE_NUM_EXPERTS=256
DEFS_dispatch_gemm_warp_specialization := -DTK_MOE_H=7168 -DTK_MOE_I=2048 -DTK_MOE_TOP_K=8 -DTK_MOE_NUM_EXPERTS=256
DEFS_ring_attention :=
DEFS_gemm_rs        :=
DEFS_dispatch_gemm_glu_combine := -DTK_MOE_H=7168 -DTK_MOE_I=2048 -DTK_MOE_TOP_K=8 -DTK_MOE_NUM_EXPERTS=256 -DTK_MOE_NUM_NODES=$(TK_MOE_NUM_NODES)

# === Build targets ===
BUILD := build
SRC   := src

KERNELS := dispatch_gemm gemm_rs ag_gemm gemm_ar ring_attention dispatch_gemm_glu_combine

SPECIALIZATION ?= sm
ifeq ($(SPECIALIZATION),warp)
BLACKWELL_SPECIALIZATION_KERNEL := dispatch_gemm_warp_specialization
else ifeq ($(SPECIALIZATION),sm)
BLACKWELL_SPECIALIZATION_KERNEL := dispatch_gemm_blackwell
else
$(error Unknown SPECIALIZATION=$(SPECIALIZATION). Use SPECIALIZATION=warp or SPECIALIZATION=sm.)
endif

ENABLE_DISPATCH_GEMM_BLACKWELL ?= 0
ifeq ($(ENABLE_DISPATCH_GEMM_BLACKWELL),1)
KERNELS += $(BLACKWELL_SPECIALIZATION_KERNEL)
endif
all: $(addprefix $(BUILD)/lib,$(addsuffix .so,$(KERNELS)))

dispatch-gemm-blackwell: $(BUILD)/lib$(BLACKWELL_SPECIALIZATION_KERNEL).so

dispatch-gemm-sm-specialization: $(BUILD)/libdispatch_gemm_blackwell.so

dispatch-gemm-warp-specialization: $(BUILD)/libdispatch_gemm_warp_specialization.so

BLACKWELL_BENCH_ARGS ?= --check
run-dispatch-gemm-blackwell: dispatch-gemm-blackwell
	$(PYTHON) -m torch.distributed.run --standalone \
	    --nproc-per-node=$(INTRA_NUM_DEVICES) \
	    bench/dispatch_gemm_blackwell_bench.py \
	        --specialization $(SPECIALIZATION) $(BLACKWELL_BENCH_ARGS)

DISPATCH_GEMM_BLACKWELL_HEADERS := \
	include/operators/dispatch_gemm_blackwell/dispatch_gemm_blackwell.cuh \
	include/operators/dispatch_gemm_blackwell/session.cuh

$(BUILD)/libdispatch_gemm_blackwell.so: $(DISPATCH_GEMM_BLACKWELL_HEADERS)

DISPATCH_GEMM_WARP_SPECIALIZATION_HEADERS := \
	include/operators/dispatch_gemm_warp_specialization/dispatch_gemm_warp_specialization.cuh \
	include/operators/dispatch_gemm_warp_specialization/session.cuh

$(BUILD)/libdispatch_gemm_warp_specialization.so: $(DISPATCH_GEMM_WARP_SPECIALIZATION_HEADERS)

$(BUILD)/lib%.so: $(SRC)/%.cu Makefile | $(BUILD)
	$(NVCC) $(COMMON_FLAGS) $(COMMON_DEFINES) -DTORCH_EXTENSION_NAME=mkernel_release_$* $(DEFS_$*) $(COMMON_INC) \
	    --compiler-options '-fPIC' $(LDFLAGS) $< -o $@

# Single-GPU microbenchmark for the already-ready acquire-load path in
# ag_gemm_kda_mla. It deliberately does not link either internode backend.
acquire-load-pass-bench: $(BUILD)/libacquire_load_pass_bench.so

$(BUILD)/libacquire_load_pass_bench.so: $(SRC)/acquire_load_pass_bench.cu \
		include/comm/atomic_u32.cuh Makefile | $(BUILD)
	$(NVCC) $(COMMON_FLAGS) $(ARCH_DEFINES) \
	    -DTORCH_EXTENSION_NAME=mkernel_release_acquire_load_pass_bench \
	    $(INC_RELEASE) $(TORCH_INC) $(PY_INC) --compiler-options '-fPIC' \
	    -shared -lcuda -L$(TORCH_LIB) -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
	    -ltorch_python -Xlinker -rpath -Xlinker $(TORCH_LIB) -L$(CUDA_HOME)/lib \
	    $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

bench: all
	cd bench && bash run.sh all bench

check: all
	cd bench && bash run.sh all check

# Host-only unit test for internode slot math (peer_rank_for_slot,
# slot_at_peer, ring origin). Pins down the N>2 invariants without needing
# real multi-node hardware.
test-slot-math: tests/test_internode_slot_math.cpp | $(BUILD)
	g++ -std=c++17 -O2 -I include -D__host__= -D__device__= $< -o $(BUILD)/test_internode_slot_math
	$(BUILD)/test_internode_slot_math

plots:
	cd plots && python3 plot_tflops_efa.py

.PHONY: all dispatch-gemm-blackwell dispatch-gemm-sm-specialization \
	dispatch-gemm-warp-specialization run-dispatch-gemm-blackwell \
	gemm-ar-blackwell run-gemm-ar-blackwell clean bench check \
	test-slot-math plots ag-gemm-kda-mla run-ag-gemm-kda-mla

run-gemm-ar-blackwell : gemm_ar_blackwell
	python -m torch.distributed.run --standalone --nproc-per-node=$(INTRA_NUM_DEVICES) bench/gemm_ar_blackwell_bench.py

gemm-ar-blackwell : $(BUILD)/libgemm_ar_blackwell.so

$(BUILD)/libgemm_ar_blackwell.so : $(SRC)/gemm_ar_blackwell.cu | $(BUILD)
	$(NVCC) $(COMMON_FLAGS) $(GEMM_AR_BLACKWELL_SANITIZE) -lineinfo --ptxas-options=-v $(COMMON_DEFINES) -DTORCH_EXTENSION_NAME=mkernel_release_gemm_ar_blackwell $(DEFS_gemm_ar_blackwell) $(COMMON_INC) \
	    --compiler-options '-fPIC' $(LDFLAGS) $< -o $@

run-ag-gemm-kda-mla : ag-gemm-kda-mla
	python -m torch.distributed.run --standalone --nproc-per-node=$(INTRA_NUM_DEVICES) bench/ag_gemm_kda_mla_bench.py

ag-gemm-kda-mla : $(BUILD)/libag_gemm_kda_mla.so

$(BUILD)/libag_gemm_kda_mla.so : $(SRC)/ag_gemm_kda_mla.cu | $(BUILD)
	$(NVCC) $(COMMON_FLAGS) $(GEMM_AR_BLACKWELL_SANITIZE) -lineinfo --ptxas-options=-v $(COMMON_DEFINES) -DTORCH_EXTENSION_NAME=mkernel_release_ag_gemm_kda_mla $(DEFS_gemm_ar_blackwell) $(COMMON_INC) \
	    --compiler-options '-fPIC' $(LDFLAGS) $< -o $@
# === In-kernel timing profile ===
#
# Same source, built with -DPROFILE_TIMINGS into a *separate* .so and module
# name, so a profile build never shadows the shipping one -- keep both around
# and let the bench pick. Without the flag every emit, the event enum and the
# TimingRecord* on fused_globals preprocess away, so the shipping cubin carries
# no profiling instructions at all.
#
# EVENTS_PER_BLOCK is the per-CTA cap on emits. The ring costs
# NUM_BLOCKS * EVENTS_PER_BLOCK * 16B (148 * 65536 * 16B = 155 MB at the
# default). Bump it if the trace shows every CTA truncating at the same x --
# M=32768 needs 131072.
EVENTS_PER_BLOCK ?= 65536

GEMM_AR_BLACKWELL_HEADERS := \
	include/operators/gemm_ar/gemm_ar_blackwell.cuh \
	include/operators/gemm_ar/gemm_ar_blackwell_session.cuh \
	include/common/timings.cuh

$(BUILD)/libgemm_ar_blackwell.so : $(GEMM_AR_BLACKWELL_HEADERS)

gemm-ar-blackwell-profile : $(BUILD)/libgemm_ar_blackwell_profile.so

$(BUILD)/libgemm_ar_blackwell_profile.so : $(SRC)/gemm_ar_blackwell.cu \
		$(GEMM_AR_BLACKWELL_HEADERS) Makefile | $(BUILD)
	$(NVCC) $(COMMON_FLAGS) -lineinfo --ptxas-options=-v $(COMMON_DEFINES) \
	    -DPROFILE_TIMINGS -DMKERNEL_EVENTS_PER_BLOCK=$(EVENTS_PER_BLOCK) \
	    -DTORCH_EXTENSION_NAME=mkernel_release_gemm_ar_blackwell_profile \
	    $(DEFS_gemm_ar_blackwell) $(COMMON_INC) \
	    --compiler-options '-fPIC' $(LDFLAGS) $< -o $@

# The profile drivers spawn their own ranks, so these are just `python ...`.
PROFILE_ARGS ?=
run-gemm-ar-blackwell-profile : gemm-ar-blackwell-profile
	$(PYTHON) bench/gemm_ar_blackwell_profile.py $(PROFILE_ARGS)

AG_GEMM_KDA_MLA_HEADERS := \
	include/operators/ag_gemm/ag_gemm_kda_mla.cuh \
	include/operators/ag_gemm/ag_gemm_kda_mla_session.cuh \
	include/common/timings.cuh

$(BUILD)/libag_gemm_kda_mla.so : $(AG_GEMM_KDA_MLA_HEADERS)

ag-gemm-kda-mla-profile : $(BUILD)/libag_gemm_kda_mla_profile.so

$(BUILD)/libag_gemm_kda_mla_profile.so : $(SRC)/ag_gemm_kda_mla.cu \
		$(AG_GEMM_KDA_MLA_HEADERS) Makefile | $(BUILD)
	$(NVCC) $(COMMON_FLAGS) -lineinfo --ptxas-options=-v $(COMMON_DEFINES) \
	    -DPROFILE_TIMINGS -DMKERNEL_EVENTS_PER_BLOCK=$(EVENTS_PER_BLOCK) \
	    -DTORCH_EXTENSION_NAME=mkernel_release_ag_gemm_kda_mla_profile \
	    $(DEFS_gemm_ar_blackwell) $(COMMON_INC) \
	    --compiler-options '-fPIC' $(LDFLAGS) $< -o $@

run-ag-gemm-kda-mla-profile : ag-gemm-kda-mla-profile
	$(PYTHON) bench/ag_gemm_kda_mla_profile.py $(PROFILE_ARGS)

# === Nsight Compute ===
#
# Deliberately built on the *shipping* .so, not the profile one: -DPROFILE_TIMINGS
# adds emits, registers and ring stores that every ncu counter would then include.
# The driver enforces this -- it refuses to run ncu against an instrumented build.
#
# NCU_REPLAY=application (the default) is not a preference: kernel replay
# snapshots every allocation reachable from the context before each pass, and the
# multicast/peer-imported DistBuffer mappings cannot be copied, so ncu dies with
# ContextSaveFailed before the first pass. Application replay re-runs instead of
# saving -- which relaunches the whole job once per metric pass, so ncu wraps
# torchrun and attaches to every rank, and cudaProfilerStart on NCU_RANK alone
# decides who actually records.
#
# Passes cost a full job restart each, hence NCU_SET=detailed rather than full.
#
#   make -j 10 GPU=blackwell run-ag-gemm-kda-mla-ncu
#   make run-ag-gemm-kda-mla-ncu NCU=/usr/local/cuda-13.1/bin/ncu NCU_SET=basic
#   make run-ag-gemm-kda-mla-ncu NCU_EXTRA='--ncu-arg=--metrics=sm__cycles_active.avg'
#
# NCU_EXTRA values that start with a dash need the = form shown above.
NCU              ?= ncu
NCU_SET          ?= detailed
NCU_REPLAY       ?= application
NCU_RANK         ?= 0
NCU_ITERS        ?= 1
NCU_EXTRA        ?=
# Shared by both ncu rules on purpose: a report is only comparable against
# another at the same shape. NCU_OUT empty lets the driver pick a per-impl
# default, so the two reports never land on the same path.
NCU_SHAPE        ?= 32768
NCU_OUT          ?=

NCU_FLAGS = --ncu-bin $(NCU) --ncu-set $(NCU_SET) --ncu-replay $(NCU_REPLAY) \
	    --ncu-rank $(NCU_RANK) --ncu-iters $(NCU_ITERS) --shape $(NCU_SHAPE) \
	    $(if $(NCU_OUT),--ncu-out $(NCU_OUT))

run-ag-gemm-kda-mla-ncu : ag-gemm-kda-mla
	$(PYTHON) bench/ag_gemm_kda_mla_profile.py --ncu $(NCU_FLAGS) \
	    $(NCU_EXTRA) $(PROFILE_ARGS)

# === CUTLASS baseline, same shape, second report ===
#
# CUTLASS is JIT-compiled per config and application replay restarts the job
# once per pass, so tuning is a separate step: it caches the bench script's
# autotuned winner to traces/.cutlass_tiler_rank<N>.json and the profiled run
# reads it. Re-tune with CUTLASS_RETUNE=1, or pin one with CUTLASS_TILER=256x256.
#
# Needs CUTLASS_PATH (or CUTLASS_AG_GEMM) set, exactly as the bench does.
#
#   make run-cutlass-ag-gemm-ncu NCU=/opt/nvidia/nsight-compute/2026.2.0/ncu
CUTLASS_TILER    ?= auto
CUTLASS_RETUNE   ?=

tune-cutlass-ag-gemm :
	$(PYTHON) bench/ag_gemm_kda_mla_profile.py --impl cutlass --cutlass-tune-only \
	    --shape $(NCU_SHAPE) --cutlass-tiler $(CUTLASS_TILER) \
	    $(if $(CUTLASS_RETUNE),--cutlass-retune)

# Depends on the tune step so one command is always correct: with the winner
# already cached that step is just a job start, not another autotune.
run-cutlass-ag-gemm-ncu : tune-cutlass-ag-gemm
	$(PYTHON) bench/ag_gemm_kda_mla_profile.py --ncu --impl cutlass $(NCU_FLAGS) \
	    --cutlass-tiler $(CUTLASS_TILER) $(NCU_EXTRA) $(PROFILE_ARGS)

.PHONY: acquire-load-pass-bench \
	gemm-ar-blackwell-profile run-gemm-ar-blackwell-profile \
	ag-gemm-kda-mla-profile run-ag-gemm-kda-mla-profile \
	run-ag-gemm-kda-mla-ncu tune-cutlass-ag-gemm run-cutlass-ag-gemm-ncu
