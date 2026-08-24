# Makefile — single-config build for 5 multinode kernels
#
# Usage:
#   make all       — build all 5 .so's into build/
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

# TEMP: undefine the BACKEND LIBS for now
undefine BACKEND_DEFINES
undefine BACKEND_LIBS

# === Target GPU ===
#   GPU=hopper    → sm_90a, wgmma MMA path (default, upstream behaviour)
#   GPU=blackwell → sm_103a, tcgen05 MMA path (B300; gemm_rs only so far)
GPU ?= hopper
ifeq ($(GPU),blackwell)
    ARCH              := -gencode arch=compute_103a,code=sm_103a
    ARCH_DEFINES      := -DKITTENS_SM10X -DKITTENS_BLACKWELL -DMKERNEL_TCGEN05
    DEFAULT_CUDA_HOME := /usr/local/cuda-13.2
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

# COMP_SM_SWEEP=1 compiles every comp/comm SM split in
# GEMM_AR_FOR_EACH_COMP_SM instead of just the default. Each extra split is a
# full set of kernel instantiations (x2 signal strategies x3 shape buckets), so
# this multiplies gemm_ar_blackwell compile time by roughly the list length --
# keep it off for normal iteration and turn it on to run the sweep.
# The two sweep axes are independent so you can vary one at a time; enabling
# both compiles their cross product (2 SW x 3 unroll x 9 split x 2 strategy =
# 108 kernels vs 8 for a default build), which is a long compile.
COMP_SM_SWEEP   ?= 0
UNROLL_SWEEP    ?= 0
SIGNAL_DEPTH_SWEEP ?= 0
SWEEP_DEFINES   :=
ifeq ($(COMP_SM_SWEEP),1)
    SWEEP_DEFINES += -DGEMM_AR_COMP_SM_SWEEP
endif
ifeq ($(UNROLL_SWEEP),1)
    SWEEP_DEFINES += -DGEMM_AR_UNROLL_SWEEP
endif
ifeq ($(SIGNAL_DEPTH_SWEEP),1)
    SWEEP_DEFINES += -DGEMM_AR_SIGNAL_DEPTH_SWEEP
endif

# EXTRA_DEFINES lets a target pin individual sweep axes, e.g.
#   EXTRA_DEFINES='-DGEMM_AR_FOR_EACH_UNROLL(F)=F(16)'
EXTRA_DEFINES   ?=
COMMON_DEFINES  := $(ARCH_DEFINES) -DINTRA_NUM_DEVICES=$(INTRA_NUM_DEVICES) $(BACKEND_DEFINES) $(SWEEP_DEFINES) $(EXTRA_DEFINES)
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
DEFS_ring_attention :=
DEFS_gemm_rs        :=
DEFS_dispatch_gemm_glu_combine := -DTK_MOE_H=7168 -DTK_MOE_I=2048 -DTK_MOE_TOP_K=8 -DTK_MOE_NUM_EXPERTS=256 -DTK_MOE_NUM_NODES=$(TK_MOE_NUM_NODES)

# === Build targets ===
BUILD := build
SRC   := src

KERNELS := dispatch_gemm gemm_rs ag_gemm gemm_ar ring_attention dispatch_gemm_glu_combine

all: $(addprefix $(BUILD)/lib,$(addsuffix .so,$(KERNELS)))

$(BUILD)/lib%.so: $(SRC)/%.cu | $(BUILD)
	$(NVCC) $(COMMON_FLAGS) $(COMMON_DEFINES) -DTORCH_EXTENSION_NAME=mkernel_release_$* $(DEFS_$*) $(COMMON_INC) \
	    --compiler-options '-fPIC' $(LDFLAGS) $< -o $@

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

.PHONY: all clean bench check test-slot-math plots sweep_comp_sm sweep_unroll sweep_depth sweep_depth_x_split sweep_unroll_x_split sweep_all

run_gemm_ar_blackwell : gemm_ar_blackwell
	python -m torch.distributed.run --standalone --nproc-per-node=$(INTRA_NUM_DEVICES) bench/gemm_ar_blackwell_bench.py

# Sweep targets. The bench picks up whatever was compiled via
# compiled_comp_sm_splits() / compiled_ar_unrolls(), so no list is duplicated.
# Prefer sweeping one axis at a time: sweep_all is the 108-kernel cross product.
sweep_comp_sm : ; $(MAKE) COMP_SM_SWEEP=1 run_gemm_ar_blackwell
sweep_unroll  : ; $(MAKE) UNROLL_SWEEP=1 run_gemm_ar_blackwell
sweep_depth   : ; $(MAKE) SIGNAL_DEPTH_SWEEP=1 run_gemm_ar_blackwell

# Targeted experiment: signal depth 0 vs 1, across every comp/comm split, with
# unroll pinned to 16 and PULL only (the pairing that has been winning). Pinning
# the settled axes keeps this at 44 kernels instead of the 528 of sweep_all.
# NOTE on quoting: the -D values contain parentheses and a space, so each one
# needs single quotes that survive to the nvcc command line. The outer double
# quotes are eaten by the shell running this recipe, leaving make holding the
# single-quoted strings; those are then stripped by the shell running nvcc.
# Single outer quotes would expose the bare "(F)" to the shell and fail.
sweep_depth_x_split :
	$(MAKE) COMP_SM_SWEEP=1 \
	        EXTRA_DEFINES="'-DGEMM_AR_FOR_EACH_UNROLL(F)=F(16)' '-DGEMM_AR_FOR_EACH_SIGNAL_DEPTH(F)=F(0) F(1)' -DGEMM_AR_ENABLE_PUSH=0" \
	        run_gemm_ar_blackwell
# Targeted experiment: the unroll matrix at the three splits that survived the
# split sweep (140/136/132), with depth 0 vs 1 and PULL only. 72 kernels.
# Unroll tops out at 43 by construction -- see the ceiling note in the header.
sweep_unroll_x_split :
	$(MAKE) COMP_SM_SWEEP=1 \
	        EXTRA_DEFINES="'-DGEMM_AR_FOR_EACH_COMP_SM(F)=F(140) F(136) F(132)' '-DGEMM_AR_FOR_EACH_UNROLL(F)=F(8) F(11) F(14) F(16) F(22) F(32) F(43)' '-DGEMM_AR_FOR_EACH_SIGNAL_DEPTH(F)=F(0) F(1)' -DGEMM_AR_ENABLE_PUSH=0" \
	        run_gemm_ar_blackwell

# 528 kernels. Prefer one axis at a time unless you are chasing an interaction.
sweep_all     : ; $(MAKE) COMP_SM_SWEEP=1 UNROLL_SWEEP=1 SIGNAL_DEPTH_SWEEP=1 run_gemm_ar_blackwell

gemm_ar_blackwell : $(BUILD)/libgemm_ar_blackwell.so

# -fdevice-sanitize=memcheck instruments EVERY device memory access. That is
# fine for the tcgen05/TMA compute path but ruinous for the multimem all-reduce
# loop, which is nothing but discrete global accesses — it turns the AR into
# the entire kernel and makes it immune to pipelining. Opt in explicitly with
# `make SANITIZE=1 gemm_ar_blackwell` when chasing a memory bug; never
# benchmark a sanitized build.
SANITIZE ?= 0
ifeq ($(SANITIZE),1)
GEMM_AR_BLACKWELL_SANITIZE := -fdevice-sanitize=memcheck
endif

$(BUILD)/libgemm_ar_blackwell.so : $(SRC)/gemm_ar_blackwell.cu | $(BUILD)
	$(NVCC) $(COMMON_FLAGS) $(GEMM_AR_BLACKWELL_SANITIZE) -lineinfo --ptxas-options=-v $(COMMON_DEFINES) -DTORCH_EXTENSION_NAME=mkernel_release_gemm_ar_blackwell $(DEFS_gemm_ar_blackwell) $(COMMON_INC) -I/home/uccl/shawn/ThunderKittens/include \
	    --compiler-options '-fPIC' $(LDFLAGS) $< -o $@