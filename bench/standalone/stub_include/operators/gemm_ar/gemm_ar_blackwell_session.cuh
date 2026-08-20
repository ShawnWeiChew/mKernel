#pragma once
// Deliberately empty.
//
// src/gemm_ar_blackwell.cu ends with
//     #include "operators/gemm_ar/gemm_ar_blackwell_session.cuh"
// which normally expands to a PYBIND11_MODULE. The standalone bench includes
// that same .cu to get the real kernel, but wants an executable, not a Python
// extension. Putting this directory ahead of include/ on the -I line shadows
// the real session header so the pybind block never appears.
//
// Quoted-include resolution: the directive lives in src/, src/operators/... does
// not exist, so the search falls through to -I order — this stub first.
